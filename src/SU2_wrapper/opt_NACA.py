from openmdao.main.api import Component, Variable, Assembly, set_as_top
from openmdao.main.datatypes.api import Array, Float
from openmdao.main.datatypes.file import File, FileRef
from openmdao.lib.drivers.api import SLSQPdriver

from SU2_wrapper import Deform, Solve, Config


class OptModel(Assembly):

    def configure(self):

        myConfig = Config()

        self.add('deform', Deform())
        self.add('solve', Solve())

        myConfig.read('inv_NACA0012.cfg')
        self.deform.config_in = myConfig

        self.connect('deform.mesh_file', 'solve.mesh_file')
        self.connect('deform.config_out', 'solve.config_in')
 
        self.add('driver', SLSQPdriver())
        self.driver.workflow.add(['deform', 'solve'])

        for j in range(38):
            self.driver.add_parameter('deform.dv_vals[%d]' % j, low=-.05, high=.05)

        self.driver.add_objective('solve.DRAG*0.001')
        self.driver.add_constraint('solve.LIFT * 0.001 > .328188 * 0.001')

if __name__ == '__main__':
    model = set_as_top(OptModel())

    model.run()

    print 'design values'
    print model.deform.dv_vals

    print 'Lift, Drag'
    print model.solve.LIFT, model.solve.DRAG