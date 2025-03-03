import os
import sys
import yaml
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import center_of_mass

CURRENT_FILE = Path(os.path.abspath(__file__))
sys.path.append(str(CURRENT_FILE.parent.parent.parent))
from main.utils import *

class Generator_Base():
    def __init__(self, params, verbose=1):
        self.input_folder = params.get("input_folder")
        self.output_folder = params.get("output_folder")
        self.verbose = verbose
        self.nbr_backgrounds = params.get("nbr_backgrounds")
        self._validate_params()
        self.existing_out_subjects = os.listdir(self.output_folder)
        self._log(f"\nInitialized generator with parameters {params}.\n")
    
    def _validate_params(self):
        if self.input_folder is not None and not os.path.exists(self.input_folder):
            raise ValueError("Please provide an existing input folder.")
        if self.output_folder is None:
            raise ValueError("Please provide an output folder.")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        if self.nbr_backgrounds is None or type(self.nbr_backgrounds) != int or self.nbr_backgrounds <= 0:
            raise ValueError("Please provide a positive number of backgrounds to generate.")
        
    def _get_next_subject_nbr(self):
        if len(self.existing_out_subjects) == 0:
            return 1
        else:
            return max([int(f.split("_")[1]) for f in self.existing_out_subjects]) + 1
        
    def generate_backgrounds(self):
        self.t0 = time.time()
        self._log(f"Generating {self.nbr_backgrounds} phantom backgrounds...\n")
        self._generate_backgrounds()
        self._log(f"\nGenerated {self.nbr_backgrounds} phantom backgrounds in {nice_time(time.time() - self.t0)}.\n")

    def _generate_backgrounds(self):
        raise NotImplementedError("Please implement a method to generate backgrounds")
    
    def _log(self, message, level=1):
        if self.verbose >= level:
            print(message)

    def _get_input_file_instance(self, iter=None, random=True):
        """
        Get the next input file to generate a background from.
        :param iter: int, iteration number.
        :param random: bool, whether to choose a random file.
        :return: str, file name.
        """
        if random:
            return np.random.choice(self.input_files)
        else:
            if iter is None:
                raise ValueError("Please provide an iteration number.")
            if iter >= len(self.input_files):
                raise ValueError(f"Please provide an iteration number less than {len(self.input_files)}.")
            return self.input_files[iter]


class Generator_XCAT(Generator_Base):
    def __init__(self, params, verbose=1):
        self.input_folder = params.get("input_folder")
        self.lv_tissue_kd = params.get("lv_tissue_kd")
        self.final_img_size = params.get("final_img_size")
        self.xcat_act_img_size = params.get("xcat_act_img_size")
        super().__init__(params, verbose)
        self.input_files = sorted(os.listdir(self.input_folder))
    
    def _validate_params(self):
        super()._validate_params()
        if self.input_folder is None:
            raise ValueError("Please provide an input folder with existing XCAT act files.")
        if self.lv_tissue_kd is None or type(self.lv_tissue_kd) != int or self.lv_tissue_kd <= 0:
            raise ValueError("Please provide a valid tissue id for the left ventricle.")
        if self.xcat_act_img_size is None or type(self.xcat_act_img_size) != int or self.xcat_act_img_size <= 0:
            raise ValueError("Please provide a valid XCAT image size.")
        if self.final_img_size is None or type(self.final_img_size) != int or self.final_img_size <= 0:
            raise ValueError("Please provide a valid final image size.")
        
    def _load_tissues(self, input_file, xcat_img_size):
        """
        Load tissue segmentations from an XCAT act file. 
        :param input_file: str, file name. Only .bin files are supported.
        :param xcat_img_size: int, size of the XCAT image.
        :return: tissue segmentation
        """
        with open(os.path.join(self.input_folder, input_file), 'rb') as fid:
            data_array = np.fromfile(fid, np.float32)
        backgrounds = data_array.reshape(-1, xcat_img_size, xcat_img_size)[0]
        return backgrounds
    

    def _process_img_crop(self, img, centre, array_size):
        """
        Crop an image around a centre point.
        :param img: np.array, image to crop.
        :param centre: tuple, centre of the crop.
        :param array_size: int, size of the cropped image.
        :return: np.array, cropped image.
        """
        start_x = centre[0] - array_size // 2
        if start_x < 0:
            start_x = 0
            array_size = centre[0]*2
            return self._process_img_crop(img, centre, array_size)
        start_y = centre[1] - array_size // 2
        if start_y < 0:
            start_y = 0
            array_size = centre[1]*2
            return self._process_img_crop(img, centre, array_size)
        end_x = start_x + array_size
        if end_x >= img.shape[-1]:
            end_x = img.shape[-1] - 1
            array_size = end_x - start_x
            return self._process_img_crop(img, centre, array_size)
        end_y = start_y + array_size
        if end_y >= img.shape[-1]:
            end_y = img.shape[-1] - 1
            array_size = end_y - start_y
            return self._process_img_crop(img, centre, array_size)
        res = img[start_x:end_x, start_y:end_y]
        return res


    def _process_tissues(self, backgrounds, crop_size, lv_tissue_id=1):
        """
        Process the tissue segmentations. Return a cropped version around the left ventricle.
        :param backgrounds: np.array, tissue segmentations.
        :param crop_size: int, size of the cropped image.
        :param lv_tissue_id: int, tissue id of the left ventricle.
        :return: np.array, processed tissue segmentations.
        """
        
        LV_centre = center_of_mass(backgrounds == lv_tissue_id)
        LV_centre_round = np.round(LV_centre).astype(int)
        backgrounds = self._process_img_crop(backgrounds, LV_centre_round, crop_size)
        
        if backgrounds.shape[-1] != crop_size:
            print("Careful! Processed data had to be reduced to size {}, as {} was too large.".format(backgrounds.shape[-1], crop_size))

        mask = (backgrounds == lv_tissue_id)

        return backgrounds, mask


    def _generate_backgrounds(self):
        if self.nbr_backgrounds == len(self.input_files):
            self._log("Number of backgrounds to generate is equal to the number of input files. Using all input files.")
            random = False
        else:
            self._log("Number of backgrounds to generate is different to the number of input files. Will randomly choose input files.")
            random = True

        for i in range(self.nbr_backgrounds):
            input_file = self._get_input_file_instance(iter=i, random=random)
            background = self._load_tissues(input_file, self.xcat_act_img_size)
            background, mask = self._process_tissues(background, self.final_img_size, self.lv_tissue_kd)

            subject_nbr = self._get_next_subject_nbr() + i
            subject_folder = os.path.join(self.output_folder, f"subject_{subject_nbr}")
            os.makedirs(subject_folder, exist_ok=True)
            np.save(os.path.join(subject_folder, f"mask.npy"), mask)
            np.save(os.path.join(subject_folder, f"tissues_seg.npy"), background)

            config = {
                "input_file": os.path.basename(input_file),
                "lv_tissue_id": self.lv_tissue_kd,
                "xcat_act_img_size": self.xcat_act_img_size,
                "final_img_size": self.final_img_size
            }
            with open(os.path.join(subject_folder, "config.yaml"), 'w') as f:
                yaml.dump(config, f)

            self._log(f"Generated subject_{subject_nbr} from input file {os.path.basename(input_file)}. Time: {nice_time(time.time() - self.t0)}")
        


class Generator_Invivo(Generator_Base):
    def __init__(self, params, verbose=1):
        self.input_folder = params.get("input_folder")
        super().__init__(params, verbose)
        self.input_files = sorted(os.listdir(self.input_folder))
    
    def _validate_params(self):
        super()._validate_params()
        if self.input_folder is None:
            raise ValueError("Please provide an input folder with existing invivo contours.")
        
    def _gen_invivo_mask(self, input_file):
        """
        Generate a binary mask from an invivo contour file.
        :param input_file: str, file name. Only .nii.gz files are supported. Assumes 2D+time input mask.
        :return: np.array, binary mask.
        """
        mask = nib.load(os.path.join(self.input_folder, input_file)).get_fdata()
        mask = mask[..., 0].astype(np.bool)
        return mask
        
    def _generate_backgrounds(self):
        if self.nbr_backgrounds == len(self.input_files):
            self._log("Number of backgrounds to generate is equal to the number of input files. Using all input files.")
            random = False
        else:
            self._log("Number of backgrounds to generate is different to the number of input files. Will randomly choose input files.")
            random = True

        for i in range(self.nbr_backgrounds):
            input_file = self._get_input_file_instance(iter=i, random=random)
            mask = self._gen_invivo_mask(input_file)

            subject_nbr = self._get_next_subject_nbr() + i
            subject_folder = os.path.join(self.output_folder, f"subject_{subject_nbr}")
            os.makedirs(subject_folder, exist_ok=True)
            np.save(os.path.join(subject_folder, f"mask.npy"), mask)

            config = {
                "input_file": os.path.basename(input_file)
            }
            with open(os.path.join(subject_folder, "config.yaml"), 'w') as f:
                yaml.dump(config, f)

            self._log(f"Generated subject_{subject_nbr} from input file {os.path.basename(input_file)}. Time: {nice_time(time.time() - self.t0)}")


class Generator_Phantom(Generator_Base):
    def __init__(self, params, verbose=1):
        self.image_size = params.get("image_size")
        self.in_radius = params.get("in_radius")
        self.out_radius = params.get("out_radius")
        super().__init__(params, verbose)

    def _validate_int_list_param(self, param, param_name):
        if param is None:
            raise ValueError(f"Please provide {param_name} parameters.")
        if type(param) == list:
            if len(param) != 2:
                raise ValueError(f"If {param_name} parameter is a list, it must have two numbers (min/max size).")
            if param[0] <= 0 or param[1] <= 0:
                raise ValueError(f"{param_name} cannot be negative. Please provide an appropriate range.")
        if type(param) == int and param <= 0:
            raise ValueError(f"{param_name} cannot be a negative integer.")
        if type(param) != int and type(param) != list:
            raise ValueError(f"{param_name} must be an integer or a list of two bounds.")
        
    def _validate_float_list_param(self, param, param_name):
        if param is None:
            raise ValueError(f"Please provide {param_name} parameters.")
        if type(param) == list:
            if len(param) != 2:
                raise ValueError(f"If {param_name} parameter is a list, it must have two numbers (min/max size).")
            if param[0] <= 0 or param[1] <= 0:
                raise ValueError(f"{param_name} cannot be negative. Please provide an appropriate range.")
            if (type(param[0]) == float or type(param[1]) == float) and (param[0] < 0 or param[1] < 0 or param[0] > 1 or param[1] > 1):
                raise ValueError(f"{param_name} being a float must be in the range (0, 1).")
        if type(param) in [float, int] and param <= 0:
            raise ValueError(f"{param_name} cannot be negative.")
        if type(param) == float and (param < 0 or param > 1):
            raise ValueError(f"{param_name} being a float must be in the range (0, 1).")
        if type(param) not in [float, int, list]:
            raise ValueError(f"{param_name} must be a float/int or a list of two bounds.")
    
    def _validate_params(self):
        super()._validate_params()
        self._validate_int_list_param(self.image_size, "image_size")
        self._validate_float_list_param(self.in_radius, "in_radius")
        self._validate_int_list_param(self.out_radius, "out_radius")
    
    def _generate_backgrounds(self):
        """
        Generate phantom masks. 
        """
        for i in range(self.nbr_backgrounds):
            image_size, in_radius, out_radius = self.gen_case_params()
            mask = self._gen_phantom_mask(image_size, in_radius, out_radius)

            subject_nbr = self._get_next_subject_nbr() + i
            subject_folder = os.path.join(self.output_folder, f"subject_{subject_nbr}")
            os.makedirs(subject_folder, exist_ok=True)
            np.save(os.path.join(subject_folder, f"mask.npy"), mask)

            config = {
                "image_size": image_size,
                "in_radius": in_radius,
                "out_radius": out_radius
            }
            with open(os.path.join(subject_folder, "config.yaml"), 'w') as f:
                yaml.dump(config, f)

            self._log(f"Generated subject_{subject_nbr} with image size {image_size}, in radius {in_radius}, out radius {out_radius}. Time: {nice_time(time.time() - self.t0)}")
    
    def gen_case_params(self):
        """
        Generate random parameters for the phantom mask.
        """
        if type(self.image_size) == list:
            image_size = np.random.randint(min(self.image_size), max(self.image_size))
        else:
            image_size = self.image_size

        if type(self.out_radius) == list:
            out_radius = np.random.randint(min(self.out_radius), min(max(self.out_radius), image_size))
        else:
            out_radius = min(self.out_radius, image_size)
        
        if type(self.in_radius) == list:
            if type(self.in_radius[0]) == int:
                in_radius = np.random.randint(min(self.in_radius), min(max(self.in_radius), out_radius))
            else:
                in_radius_ratio = np.random.uniform(min(self.in_radius), max(self.in_radius))
                in_radius = int(in_radius_ratio * out_radius)
        else:
            if type(self.in_radius) == int:
                in_radius = min(self.in_radius, out_radius)
            else:
                in_radius = int(self.in_radius * out_radius)
        
        return image_size, in_radius, out_radius


    def _gen_phantom_mask(self, image_size, in_radius, out_radius, **kwargs):
        """
        Generate a phantom mask with a annular region of interest.
        :param image_size: int, size of the image.
        :param in_radius: int, inner radius of the annular region.
        :param out_radius: int, outer radius of the annular region.
        :return: np.array, binary mask with the annular region.
        """
        y, x = np.ogrid[-image_size//2:image_size//2, -image_size//2:image_size//2]
        in_mask = x**2 + y**2 <= (in_radius//2)**2
        out_mask = x**2 + y**2 <= (out_radius//2)**2
        mask = np.zeros((image_size, image_size))
        mask[out_mask] = 1
        mask[in_mask] = 0

        return mask

