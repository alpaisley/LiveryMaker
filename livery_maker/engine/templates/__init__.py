from .toliss_a319 import ToLissA319
from .toliss_a321 import ToLissA321
from .erj145 import ERJ145
from .fjs_q400 import FJSQ400
from .xcrafts_e190 import XCraftsE190
from .jf_bae146 import JFBAe146

TEMPLATES = {t.id: t for t in (ToLissA319(), ToLissA321(), ERJ145(), FJSQ400(), XCraftsE190(), JFBAe146())}
