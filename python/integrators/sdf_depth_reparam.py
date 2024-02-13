import drjit as dr
import mitsuba as mi
from mitsuba.ad.integrators.common import mis_weight

from .reparam import ReparamIntegrator


class SdfDepthReparamIntegrator(ReparamIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.use_aovs = props.get('use_aovs', False)
        self.hide_emitters = props.get('hide_emitters', False)
        self.detach_indirect_si = props.get('detach_indirect_si', False)
        self.decouple_reparam = props.get('decouple_reparam', False)

    def sample(self, mode, scene, sampler, ray,
               δL,  state_in, reparam, active, **kwargs):
        active = mi.Mask(active)
        # Reparameterize only if we are not rendering in primal mode
        reparametrize = True and mode != dr.ADMode.Primal
        reparam_primary_ray = True and reparametrize
        si, si_d0, det, extra_output = self.ray_intersect(scene, sampler, ray, depth=0, reparam=reparam_primary_ray)
        valid_ray = (not self.hide_emitters) and scene.environment() is not None
        valid_ray |= si.is_valid()

        throughput = mi.Spectrum(1.0)
        result = mi.Spectrum(0.0)
        throughput *= det
        primary_det = det
        result += throughput * dr.select(active, si.emitter(scene, active).eval(si, active), 0.0)

        aovs = [extra_output[k] if (extra_output is not None) and (k in extra_output)
                else mi.Float(0.0) for k in self.aov_names()]
        
        # 设置一个极大值用于替换 inf
        max_t = 2

        # 使用 Dr.Jit 的条件表达式替换 inf 值
        t = dr.select(si.t >=max_t, max_t/max_t, si.t/max_t)

        return dr.select(valid_ray, mi.Spectrum(t),0.0), valid_ray, primary_det, aovs


mi.register_integrator("sdf_depth_reparam", lambda props: SdfDepthReparamIntegrator(props))
