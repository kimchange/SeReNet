from .models import register, make


try:
    from . import serenet
except:
    pass

try:
    from . import serenetsf
except:
    pass

try:
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