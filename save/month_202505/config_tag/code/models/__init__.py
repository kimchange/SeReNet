from .models import register, make
from . import serenet
from . import fserenet

try:
    from . import serenetup5
except:
    pass

try:
    from . import serenetincline
except:
    pass

try:
    from . import serenetfrv1
except:
    pass

try:
    from . import serenetfr
except:
    pass

try:
    from . import UNetDiscriminatorWithSpectralNorm
    from . import unet
except:
    pass

try:
    from . import rlfm
except:
    pass

try:
    from . import vcdnet
except:
    pass


try:
    from . import hylfmnet
except:
    pass


try:
    from . import rlnet
except:
    pass

try:
    from . import vsnet
except:
    pass