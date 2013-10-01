__all__ = ['Direct']

import os, sys, copy
import csv
import numpy as np

from openmdao.main.api import Component, Variable
from openmdao.main.datatypes.api import Array, Float
from openmdao.main.datatypes.file import File, FileRef

from SU2.io import Config, State, restart2solution
from SU2.io.tools import get_adjointSuffix, add_suffix
from SU2.run import direct, deform, adjoint, projection
from SU2.mesh.tools import read as meshread
from SU2.mesh.tools import get_markerPoints

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
    def __init__(self, default_value=None, **metadata):
        super(ConfigVar, self).__init__(default_value=default_value,
                                    	**metadata)

    def validate(self, obj, name, value):

        if not isinstance(value, Config):
	    raise TypeError("value of '%s' must be a Config object" % name)
        return value

def pts_from_mesh(meshfile, config):

    mesh = meshread(meshfile)

    markers = config.MARKER_MONITORING
    markers = markers.strip().strip('()').strip()
    markers = ''.join(markers.split())
    markers = markers.split(',')

    _,nodes = get_markerPoints(mesh,markers)

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
                sens.append((int(parts[0]), float(parts[1])))
    return [s for i,s in sorted(sens)]

# ------------------------------------------------------------
#  SU2 Suite Interface Functions
# ------------------------------------------------------------

class Deform(Component):

    config_in = ConfigVar(Config(), iotype='in')
    dv_vals = Array([], iotype='in')
    config_out = ConfigVar(Config(), iotype='out', copy='deep', data_shape=(1,))

    def _config_in_changed(self, old, new):
        meshfile = self.config_in['MESH_FILENAME']
        # - read number of unique pts from mesh file
        self.npts = pts_from_mesh(meshfile, self.config_in)
        # - create mesh_file trait with data_shape attribute
        self.add('mesh_file', File(iotype='out', data_shape=(self.npts,1)))
        self.dv_vals = np.zeros(len(self.config_in.DEFINITION_DV['KIND']))
	self.config_out = self.config_in

    def execute(self):
        # local copy
        # TODO: SU2 deform needs to be able to take an array in, too
        state = deform(self.config_in, list(self.dv_vals))
        self.mesh_file = FileRef(path=self.config_in.MESH_FILENAME)
        self.config_out = self.config_in

    def linearize(self):
        self.config_in.SURFACE_ADJ_FILENAME = self.config_in.SURFACE_FLOW_FILENAME
        projection(self.config_in)
	
        # read Jacobian info from file
	self.JT = np.zeros((len(self.dv_vals), self.npts))
	csvname = 'geo_jacobian.csv'
	with open(csvname, 'r') as infile:
	    reader = csv.DictReader(infile)
	    
	    # TODO- this is slow, rewrite it
	    
	    for j, line in enumerate(reader):
		line = {k.strip() : val for k, val in line.iteritems()}
		del line['"DesignVariable"']
	    
		vals = [val for k, val in sorted(line.iteritems())]
		self.JT[j, :] = vals

		
    def apply_derivT(self, arg, result):
	""" Matrix vector multiplication on the transposed Jacobian"""

	if 'mesh_file' in arg and 'dv_vals' in result:
	    result['dv_vals'] += self.JT.dot(arg['mesh_file']).flatten()
      
_obj_names = [
    "LIFT",
    "DRAG",
    #"SIDEFORCE",
    #"MOMENT_X",
    #"MOMENT_Y",
    #"MOMENT_Z",
    #"FORCE_X",
    #"FORCE_Y",
    #"FORCE_Z"
]

class Solve(Component):

    config_in = ConfigVar(Config(), iotype='in')
    mesh_file = File(iotype='in')
    
    def __init__(self):
        super(Solve, self).__init__()
        self.config_in = Config()
        for name in _obj_names:
            self.add_trait(name, Float(0.0, iotype='out'))

    def configure(self):
        pass

    def execute(self):
        # local copy
        state = direct(self.config_in)
	restart2solution(self.config_in, state)
        for name in _obj_names:
            setattr(self, name, state.FUNCTIONS[name])

    def linearize(self):
	""" Create jacobian from adjoint results."""
	
        self.J = None
        for i,name in enumerate(_obj_names):
            self.config_in.ADJ_OBJ_FUNC = name
            state = adjoint(self.config_in)
	    restart2solution(self.config_in, state)
            csvname = self.config_in.SURFACE_ADJ_FILENAME+'.csv'
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
    
    # need actual config file here
    myConfig = Config()
    
    model = set_as_top(Assembly())
    
    model.add('deform', Deform())
    model.add('solve', Solve())

    myConfig.read('inv_NACA0012.cfg')
    model.deform.config_in = myConfig
    
    model.connect('deform.mesh_file', 'solve.mesh_file')
    model.connect('deform.config_out', 'solve.config_in')
    
    model.driver.workflow.add(['deform', 'solve'])

    model.run()
    
    inputs = ['deform.dv_vals']
    outputs = ['solve.LIFT', 'solve.DRAG']
    J = model.driver.workflow.calc_gradient(inputs=inputs,
                                            outputs=outputs, 
                                            mode='adjoint')
    print J
    print '---'
    
    model.driver.workflow.config_changed()
    J = model.driver.workflow.calc_gradient(inputs=inputs,
                                            outputs=outputs, 
                                            fd=True)    
    print J

