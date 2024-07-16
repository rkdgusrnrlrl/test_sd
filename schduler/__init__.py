from diffusers import DPMSolverMultistepScheduler

dpm_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
dpm_sde_karras_scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True)