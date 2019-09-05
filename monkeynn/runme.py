from monkeynn import *
from dolphinn import *
import utils as fr

(D1,P)=fr.fvecs_read("siftsmall/siftsmall_base.fvecs")
(D2,Q)=fr.fvecs_read("siftsmall/siftsmall_query.fvecs")
K=int(np.log2(len(P)))-2

dol=Dolphinn(P, D1, K)
mon=monkeynn(P)
mon.whoami()