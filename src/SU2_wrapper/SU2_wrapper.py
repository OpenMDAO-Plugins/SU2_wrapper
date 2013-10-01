__all__ = ['Direct']

import os, sys, copy
import numpy as np

from openmdao.main.api import Component, Variable
from openmdao.main.datatypes.api import Array, Float
from openmdao.main.datatypes.file import File, FileRef

from SU2.io import Config, State
from SU2.run import direct, deform, adjoint, projection

# ------------------------------------------------------------
#  Setup
# ------------------------------------------------------------

SU2_RUN = os.environ['SU2_RUN'] 
sys.path.append( SU2_RUN )

# SU2 suite run command template
base_Command = os.path.join(SU2_RUN,'%s')

# check for slurm
slurm_job = os.environ.has_key('SLURM_JOBID')
    
# set mpi command
if slurm_job:
    mpi_Command = 'srun -n %i %s'
else:
    mpi_Command = 'mpirun -np %i %s'


class ConfigVar(Variable):
    def __init__(self, default_value=None, iotype=None, desc=None, 
                 **metadata):
        super(ConfigVar, self).__init__(default_value=default_value,
                                    	**metadata)

    def validate(self, obj, name, value):
        """ Validates that a specified value is valid for this trait.
        Units are converted as needed.
        """
        if not isinstance(value, Config):
        	raise TypeError("value of '%s' must be a Config object" % name)
        return value

def pts_from_mesh(meshfile, config):

    mesh = SU2.mesh.tools.read(meshfile)

    markers = config.MARKER_MONITORING
    markers = markers.strip().strip('()').strip()
    markers = ''.join(markers.split())
    markers = markers.split(',')

    _,nodes = SU2.mesh.tools.get_markerPoints(mesh,markers)

    return len(nodes)

def get_sensitivities(csvfile):
    """Return a list of sensitivities, sorted by index,
    from a csv file.
    """
    sens = []
    with open(csvfile, 'r') as f:
        for i,line in enumerate(f):
            if i > 0:
                parts = line.split(',')
                sens.append(int(parts[0]), float(parts[1]))
    return [s for i,s in sorted(sens)]

# ------------------------------------------------------------
#  SU2 Suite Interface Functions
# ------------------------------------------------------------

class Deform(Component):

    config_in = ConfigVar(Config(), iotype='in')
    dv_vals = Array([], iotype='in')
    config_out = ConfigVar(Config(), iotype='out', copy='deep')

    def _config_in_changed(self):
        meshfile = self.config_in['MESH_FILENAME']
        # - read number of unique pts from mesh file
        npts = pts_from_mesh(meshfile)
        # - create mesh_file trait with data_shape attribute
        self.add('mesh_file', File(iotype='out', data_shape=(npts,1)))
        self.dv_vals = np.zeros(len(self.config.DEFINITION_DV['KIND']))

    def execute(self):
	    # local copy
        state = deform(self.config_in)
        self.mesh_file = FileRef(path=state.FILES.MESH)
        self.config_out = self.config_in

    def linearize(self):
        self.config_in.SURFACE_ADJ_FILENAME = self.config_in.SURFACE_FLOW_FILENAME
        projection(self.config_in)
        # read Jacobian info from file

    def apply_derivT(self, arg, result):
	""" Matrix vector multiplication on the transposed Jacobian"""

	if 'mesh_file' in arg and 'dv_vals' in result:
	    result['dv_vals'] += self.J.dot(arg['mesh_file'])
      
_obj_names = [
    "LIFT",
    "DRAG",
    "SIDEFORCE",
    "MOMENT_X",
    "MOMENT_Y",
    "MOMENT_Z",
    "FORCE_X",
    "FORCE_Y",
    "FORCE_Z"
]

class Solve(Component):

    config_in = ConfigVar(Config(), iotype='in')
    mesh_file = File(iotype='in')
    
    def __init__(self):
	
	super(Solve, self).__init__()
	for name in _obj_names:
	    self.add_trait(name, Float(0.0, iotype='out'))

    def configure(self):
        pass

    def execute(self):
        # local copy
        state = direct(self.config_in)
        for name in _obj_names:
            setattr(self, name, state.FUNCTIONS[name])

    def linearize(self):
	""" Create jacobian from adjoint results."""
	
        self.J = None
        for i,name in enumerate(_obj_names):
            config_in.ADJ_OBJ_FUNC = name
            state = adjoint(self.config_in)
            csvname = self.config_in.SURFACE_ADJ_FILENAME+'.csv'
            suffix = SU2.io.tools.get_adjointSuffix(name)
            csvname = SU2.io.tools.add_suffix(csvname, suffix)
            col = get_sensitivities(csvname)
            if self.J is None:
                self.J = np.zeros((len(col),len(_obj_names)))
            self.J[:,i] = np.array(col)

    def apply_derivT(self, arg, result):
	""" Matrix vector multiplication on the transposed Jacobian"""
	
	if 'mesh_file' in result:
	    for j, name in enumerate(_obj_names):
		if name in arg:
		    result['mesh_file'] += self.J[:, j]*arg[name]
      
if __name__ == '__main__':
    from openmdao.main.api import set_as_top, Assembly
    
    myConfig = Config()
    
    model = set_as_top(Assembly())
    
    model.add('deform', Deform())
    model.add('solve', Solve())
    
    model.connect('deform.mesh_file', 'solve.mesh_file')
    model.deform.config_in.read('inv_NACA0012.cfg')
    
    model.driver.add('deform', 'solve')

    model.driver.run()
