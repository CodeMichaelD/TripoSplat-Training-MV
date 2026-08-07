import torch
import numpy as np
from plyfile import PlyData, PlyElement
from .general_utils import inverse_sigmoid, strip_symmetric, build_scaling_rotation
import utils3d

class Gaussian:
    def __init__(
            self, 
            aabb : list,
            sh_degree : int = 0,
            mininum_kernel_size : float = 0.0,
            scaling_bias : float = 0.01,
            opacity_bias : float = 0.1,
            scaling_activation : str = "exp",
            device='cuda'
        ):
        self.init_params = {
            'aabb': aabb,
            'sh_degree': sh_degree,
            'mininum_kernel_size': mininum_kernel_size,
            'scaling_bias': scaling_bias,
            'opacity_bias': opacity_bias,
            'scaling_activation': scaling_activation,
        }
        
        self.sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self.mininum_kernel_size = mininum_kernel_size 
        self.scaling_bias = scaling_bias
        self.opacity_bias = opacity_bias
        self.scaling_activation_type = scaling_activation
        self.device = device
        self.aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
        self.setup_functions()

        self._storage = {}
        # Pre-initialize keys
        for key in ['_xyz', 'xyz', '_features_dc', 'features_dc', 
                    '_features_rest', 'features_rest', 
                    '_scaling', 'scaling', 
                    '_rotation', 'rotation', 
                    '_opacity', 'opacity',
                    'features']:
            self._storage[key] = None

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        if self.scaling_activation_type == "exp":
            self.scaling_activation = torch.exp
            self.inverse_scaling_activation = torch.log
        elif self.scaling_activation_type == "softplus":
            self.scaling_activation = torch.nn.functional.softplus
            self.inverse_scaling_activation = lambda x: x + torch.log(-torch.expm1(-x))

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        
        self.scale_bias = self.inverse_scaling_activation(torch.tensor(self.scaling_bias)).to(self.device)
        self.rots_bias = torch.zeros((4)).to(self.device)
        self.rots_bias[0] = 1
        self.opacity_bias = self.inverse_opacity_activation(torch.tensor(self.opacity_bias)).to(self.device)

    def _get_store(self, name):
        return self._storage.get(name)

    def _set_store(self, name, value):
        self._storage[name] = value

    # --- Opacity ---
    @property
    def opacity(self):
        return self._get_store("opacity")

    @opacity.setter
    def opacity(self, value):
        self.set_opacity(value)

    @property
    def _opacity(self):
        return self._get_store("_opacity")

    @_opacity.setter
    def _opacity(self, value):
        self.set_raw_opacity(value)

    def set_opacity(self, value):
        if value is None:
            self._set_store("opacity", None)
            self._set_store("_opacity", None)
            return
        self._set_store("opacity", value)
        # Update raw: inverse activation
        raw = self.inverse_opacity_activation(value) - self.opacity_bias
        self._set_store("_opacity", raw)

    def set_raw_opacity(self, value):
        if value is None:
            self._set_store("_opacity", None)
            self._set_store("opacity", None)
            return
        self._set_store("_opacity", value)
        # Update activated
        activated = self.opacity_activation(value + self.opacity_bias)
        self._set_store("opacity", activated)
    
    # --- Scaling ---
    @property
    def scaling(self):
        return self._get_store("scaling")

    @scaling.setter
    def scaling(self, value):
        self.set_scaling(value)

    @property
    def _scaling(self):
        return self._get_store("_scaling")

    @_scaling.setter
    def _scaling(self, value):
        self.set_raw_scaling(value)

    def set_scaling(self, value):
        if value is None:
            self._set_store("scaling", None)
            self._set_store("_scaling", None)
            return
        self._set_store("scaling", value)
        # Inverse activation
        s = torch.sqrt(torch.square(value) - self.mininum_kernel_size ** 2)
        raw = self.inverse_scaling_activation(s) - self.scale_bias
        self._set_store("_scaling", raw)

    def set_raw_scaling(self, value):
        if value is None:
            self._set_store("_scaling", None)
            self._set_store("scaling", None)
            return
        self._set_store("_scaling", value)
        # Forward activation
        s = self.scaling_activation(value + self.scale_bias)
        s = torch.square(s) + self.mininum_kernel_size ** 2
        activated = torch.sqrt(s)
        self._set_store("scaling", activated)

    # --- Rotation ---
    @property
    def rotation(self):
        return self._get_store("rotation")

    @rotation.setter
    def rotation(self, value):
        self.set_rotation(value)

    @property
    def _rotation(self):
        return self._get_store("_rotation")

    @_rotation.setter
    def _rotation(self, value):
        self.set_raw_rotation(value)

    def set_rotation(self, value):
        if value is None:
            self._set_store("rotation", None)
            self._set_store("_rotation", None)
            return
        self._set_store("rotation", value)
        # Inverse: just subtract bias
        raw = value - self.rots_bias[None, :]
        self._set_store("_rotation", raw)

    def set_raw_rotation(self, value):
        if value is None:
            self._set_store("_rotation", None)
            self._set_store("rotation", None)
            return
        self._set_store("_rotation", value)
        # Forward activation
        activated = self.rotation_activation(value + self.rots_bias[None, :])
        self._set_store("rotation", activated)

    # --- XYZ ---
    @property
    def xyz(self):
        return self._get_store("xyz")

    @xyz.setter
    def xyz(self, value):
        self.set_xyz(value)

    @property
    def _xyz(self):
        return self._get_store("_xyz")

    @_xyz.setter
    def _xyz(self, value):
        self.set_raw_xyz(value)

    def set_xyz(self, value):
        if value is None:
            self._set_store("xyz", None)
            self._set_store("_xyz", None)
            return
        self._set_store("xyz", value)
        # Inverse
        raw = (value - self.aabb[None, :3]) / self.aabb[None, 3:]
        self._set_store("_xyz", raw)

    def set_raw_xyz(self, value):
        if value is None:
            self._set_store("_xyz", None)
            self._set_store("xyz", None)
            return
        self._set_store("_xyz", value)
        # Forward activation
        activated = value * self.aabb[None, 3:] + self.aabb[None, :3]
        self._set_store("xyz", activated)

    # --- Features (Combined) ---
    @property
    def features(self):
        return self._get_store("features")

    @features.setter
    def features(self, value):
        self.set_features(value)

    def set_features(self, value):
        self._set_store("features", value)
        if value is None:
            self._set_store("_features_dc", None)
            self._set_store("features_dc", None)
            self._set_store("_features_rest", None)
            self._set_store("features_rest", None)
            return

        # Split features into DC and Rest
        # Assuming shape (N, 1+K, 3)
        dc = value[:, :1, :]
        self._set_store("_features_dc", dc)
        self._set_store("features_dc", dc)
        
        if value.shape[1] > 1:
            rest = value[:, 1:, :]
            self._set_store("_features_rest", rest)
            self._set_store("features_rest", rest)
        else:
            self._set_store("_features_rest", None)
            self._set_store("features_rest", None)

    def _update_features_from_components(self):
        dc = self._get_store("_features_dc")
        rest = self._get_store("_features_rest")
        if dc is None:
            self._set_store("features", None)
            return
            
        if rest is not None:
            # Concatenate on dim 1 (N, 1+K, 3)
            f = torch.cat((dc, rest), dim=1)
            self._set_store("features", f)
        else:
            self._set_store("features", dc)

    # --- Features DC ---
    @property
    def features_dc(self):
        return self._get_store("features_dc")

    @features_dc.setter
    def features_dc(self, value):
        self.set_features_dc(value)

    @property
    def _features_dc(self):
        return self._get_store("_features_dc")

    @_features_dc.setter
    def _features_dc(self, value):
        self.set_raw_features_dc(value)

    def set_features_dc(self, value):
        self._set_store("features_dc", value)
        self._set_store("_features_dc", value) # Identity
        self._update_features_from_components()

    def set_raw_features_dc(self, value):
        self._set_store("_features_dc", value)
        self._set_store("features_dc", value) # Identity
        self._update_features_from_components()

    # --- Features Rest ---
    @property
    def features_rest(self):
        return self._get_store("features_rest")

    @features_rest.setter
    def features_rest(self, value):
        self.set_features_rest(value)

    @property
    def _features_rest(self):
        return self._get_store("_features_rest")

    @_features_rest.setter
    def _features_rest(self, value):
        self.set_raw_features_rest(value)

    def set_features_rest(self, value):
        self._set_store("features_rest", value)
        self._set_store("_features_rest", value)
        self._update_features_from_components()

    def set_raw_features_rest(self, value):
        self._set_store("_features_rest", value)
        self._set_store("features_rest", value)
        self._update_features_from_components()
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation + self.rots_bias[None, :])
        
    # --- Old API ---
    @property
    def get_scaling(self):
        return self._get_store("scaling")
    
    @property
    def get_rotation(self):
        return self._get_store("rotation")
    
    @property
    def get_xyz(self):
        return self._get_store("xyz")
    
    @property
    def get_features(self):
        return self.features
    
    @property
    def get_opacity(self):
        return self._get_store("opacity")
    
    # --- Saver and Loader ---
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
        
    def save_ply(self, path, transform=[[1, 0, 0], [0, 0, -1], [0, 1, 0]]):
        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = inverse_sigmoid(self.get_opacity).detach().cpu().numpy()
        scale = torch.log(self.get_scaling).detach().cpu().numpy()
        rotation = (self._rotation + self.rots_bias[None, :]).detach().cpu().numpy()
        
        if transform is not None:
            transform = np.array(transform)
            xyz = np.matmul(xyz, transform.T)
            rotation = utils3d.numpy.quaternion_to_matrix(rotation)
            rotation = np.matmul(transform, rotation)
            rotation = utils3d.numpy.matrix_to_quaternion(rotation)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path, transform=[[1, 0, 0], [0, 0, -1], [0, 1, 0]]):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        if self.sh_degree > 0:
            extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
            assert len(extra_f_names)==3*(self.sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        if transform is not None:
            transform = np.array(transform)
            xyz = np.matmul(xyz, transform)
            rotation = utils3d.numpy.quaternion_to_matrix(rots)
            rotation = np.matmul(transform.T, rotation)
            rots = utils3d.numpy.matrix_to_quaternion(rotation)
            
        # convert to actual gaussian attributes
        xyz = torch.tensor(xyz.copy(), dtype=torch.float, device=self.device)
        features_dc = torch.tensor(features_dc.copy(), dtype=torch.float, device=self.device).transpose(1, 2).contiguous()
        if self.sh_degree > 0:
            features_extra = torch.tensor(features_extra.copy(), dtype=torch.float, device=self.device).transpose(1, 2).contiguous()
        opacities = torch.sigmoid(torch.tensor(opacities.copy(), dtype=torch.float, device=self.device))
        scales = torch.exp(torch.tensor(scales.copy(), dtype=torch.float, device=self.device))
        rots = torch.tensor(rots.copy(), dtype=torch.float, device=self.device)
        
        # convert to _hidden attributes
        self._xyz = (xyz - self.aabb[None, :3]) / self.aabb[None, 3:]
        self.set_features_dc(features_dc)
        if self.sh_degree > 0:
            self.set_features_rest(features_extra)
        else:
            self.set_features_rest(None)
        self._opacity = self.inverse_opacity_activation(opacities) - self.opacity_bias
        self._scaling = self.inverse_scaling_activation(torch.sqrt(torch.square(scales) - self.mininum_kernel_size ** 2)) - self.scale_bias
        self._rotation = rots - self.rots_bias[None, :]
