"""Implements the samplers."""

import torch
from tqdm import tqdm


class EulerSampler:
    """Implements the Euler sampler for flow matching."""

    def __init__(
        self, in_resolution, in_channels, num_tokens=None, self_cond_in_dim=None
    ):
        """Constructor."""
        self.in_resolution = (
            [in_resolution, in_resolution]
            if isinstance(in_resolution, int)
            else in_resolution
        )
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.self_cond_in_dim = self_cond_in_dim

    @torch.no_grad()
    def sample(
        self,
        model,
        null_conditions,
        conditions,
        null_condition_masks=None,
        condition_masks=None,
        num_sampling_steps=250,
        cfg=1.0,
        generator=None,
        x_t=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
        return_self_condition=False,
        return_last_only=False,
        cfg_interval=[0.0, 1.0],
        use_zero_self_condition=False,
        **kwargs,
    ):
        """Samples images using the Euler method for flow matching."""

        batch_size = len(null_conditions)

        ## Setting the initial noise ##
        if x_t is None:
            x_t = torch.randn(  # Shape: (b, c, h, w)
                batch_size,
                self.in_channels,
                *self.in_resolution,
                generator=generator,
                device=device,
                dtype=dtype,
            )

        ## Setting the step size ##
        step_size = 1.0 / num_sampling_steps

        if self.self_cond_in_dim is not None:
            cond_self_condition = torch.zeros(
                batch_size,
                self.num_tokens,
                self.self_cond_in_dim,
                device=x_t.device,
                dtype=x_t.dtype,
            )
            null_self_condition = torch.zeros(
                batch_size,
                self.num_tokens,
                self.self_cond_in_dim,
                device=x_t.device,
                dtype=x_t.dtype,
            )

        else:
            cond_self_condition = None
            null_self_condition = None

        cond_self_condition_features = None
        null_self_condition_features = None

        self_condition = [
            (
                cond_self_condition.detach().cpu()
                if cond_self_condition is not None
                else None
            )
        ]
        images = [x_t.to(dtype).detach().cpu()]
        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        ):
            ## Looping over the sampling steps ##
            for step in tqdm(range(num_sampling_steps), total=num_sampling_steps):
                ## Getting the timesteps ##
                t = step / num_sampling_steps
                t_now = torch.tensor([t] * batch_size, device=device, dtype=dtype)

                cond_batch = {
                    "noisy_imgs": x_t,
                    "conditions": conditions,
                    "condition_masks": condition_masks,
                    "timesteps": t_now,
                    "self_condition": cond_self_condition,
                    "self_condition_features": cond_self_condition_features,
                }

                null_batch = {
                    "noisy_imgs": x_t,
                    "conditions": null_conditions,
                    "condition_masks": null_condition_masks,
                    "timesteps": t_now,
                    "self_condition": null_self_condition,
                    "self_condition_features": null_self_condition_features,
                }

                ## Get the prediction for conditional batch ##
                cond_batch = model(cond_batch)
                if not use_zero_self_condition:
                    cond_self_condition = (
                        cond_batch["self_condition"].detach()
                        if (
                            "self_condition" in cond_batch
                            and cond_batch["self_condition"] != None
                        )
                        else None
                    )
                    cond_self_condition_features = (
                        cond_batch["self_condition_features"].detach()
                        if (
                            "self_condition_features" in cond_batch
                            and cond_batch["self_condition_features"] != None
                        )
                        else None
                    )

                low, high = cfg_interval

                if cfg > 1.0 and (t < high and ((low == 0) | (t > low))):

                    ## Get the prediction for unconditional batch ##
                    null_batch = model(null_batch)
                    if not use_zero_self_condition:
                        null_self_condition = (
                            null_batch["self_condition"].detach()
                            if (
                                "self_condition" in null_batch
                                and null_batch["self_condition"] != None
                            )
                            else None
                        )
                        null_self_condition_features = (
                            null_batch["self_condition_features"].detach()
                            if (
                                "self_condition_features" in null_batch
                                and null_batch["self_condition_features"] != None
                            )
                            else None
                        )

                        velocity = null_batch["out"] + cfg * (
                            cond_batch["out"] - null_batch["out"]
                        )

                else:
                    velocity = cond_batch["out"]

                if return_self_condition:
                    if "diff" in cond_batch:
                        current_self_condition = (
                            cond_batch["diff"].detach().cpu()
                        )  # (b, C, h, w) — full channels kept for downstream PCA viz
                    else:
                        current_self_condition = (
                            cond_self_condition.detach().cpu()
                            if cond_self_condition is not None
                            else None
                        )
                    fake_x_0 = x_t + (1 - t) * velocity
                    current_image = fake_x_0.to(dtype).detach().cpu()

                    if return_last_only:
                        self_condition = [current_self_condition]
                        last_image = x_t + step_size * velocity
                        images = [last_image.to(dtype).detach().cpu()]
                    else:
                        self_condition.append(current_self_condition)
                        images.append(current_image)

                ## Updating the x_t using the Euler method ##
                x_t = x_t + step_size * velocity

        ## Returning the final image ##
        if return_self_condition:
            return images, self_condition
        return x_t.to(dtype), self_condition


class AutoGuidanceEulerSampler:

    def __init__(
        self, in_resolution, in_channels, num_tokens=None, self_cond_in_dim=None
    ):
        """Constructor."""
        self.in_resolution = (
            [in_resolution, in_resolution]
            if isinstance(in_resolution, int)
            else in_resolution
        )
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.self_cond_in_dim = self_cond_in_dim

    @torch.no_grad()
    def sample(
        self,
        model,
        null_conditions,
        conditions,
        null_condition_masks=None,
        condition_masks=None,
        num_sampling_steps=250,
        cfg=1.0,
        generator=None,
        x_t=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
        return_self_condition=False,
        guiding_model=None,
    ):
        """Samples images using the Euler method with AutoGuidance."""

        assert guiding_model != None, "Guiding model must be provided for AutoGuidance."

        batch_size = len(conditions)

        ## Setting the initial noise ##
        if x_t is None:
            x_t = torch.randn(  # Shape: (b, c, h, w)
                batch_size,
                self.in_channels,
                *self.in_resolution,
                generator=generator,
                device=device,
                dtype=dtype,
            )

        ## Setting the step size ##
        step_size = 1.0 / num_sampling_steps

        if self.self_cond_in_dim is not None:
            main_self_condition = torch.zeros(
                batch_size,
                self.num_tokens,
                self.self_cond_in_dim,
                device=x_t.device,
                dtype=x_t.dtype,
            )
            guiding_self_condition = torch.zeros(
                batch_size,
                self.num_tokens,
                self.self_cond_in_dim,
                device=x_t.device,
                dtype=x_t.dtype,
            )

        else:
            main_self_condition = None
            guiding_self_condition = None

        main_self_condition_features = None
        guiding_self_condition_features = None

        self_condition = [
            (
                main_self_condition.detach().cpu()
                if main_self_condition is not None
                else None
            )
        ]
        images = [x_t.to(dtype).detach().cpu()]
        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        ):
            ## Looping over the sampling steps ##
            for step in tqdm(range(num_sampling_steps), total=num_sampling_steps):
                ## Getting the timesteps ##
                t = step / num_sampling_steps
                t_now = torch.tensor([t] * batch_size, device=device, dtype=dtype)

                main_batch = {
                    "noisy_imgs": x_t,
                    "conditions": conditions,
                    "condition_masks": condition_masks,
                    "timesteps": t_now,
                    "self_condition": main_self_condition,
                    "self_condition_features": main_self_condition_features,
                }

                guiding_batch = {
                    "noisy_imgs": x_t,
                    "conditions": conditions,
                    "condition_masks": condition_masks,
                    "timesteps": t_now,
                    "self_condition": guiding_self_condition,
                    "self_condition_features": guiding_self_condition_features,
                }

                ## Get the prediction from the main model ##
                main_batch = model(main_batch)
                main_self_condition = (
                    main_batch["self_condition"].detach()
                    if (
                        "self_condition" in main_batch
                        and main_batch["self_condition"] != None
                    )
                    else None
                )
                main_self_condition_features = (
                    main_batch["self_condition_features"].detach()
                    if (
                        "self_condition_features" in main_batch
                        and main_batch["self_condition_features"] != None
                    )
                    else None
                )

                if cfg > 1.0:

                    ## Get the prediction from the (weaker) guiding model ##
                    guiding_batch = guiding_model(guiding_batch)
                    guiding_self_condition = (
                        guiding_batch["self_condition"].detach()
                        if (
                            "self_condition" in guiding_batch
                            and guiding_batch["self_condition"] != None
                        )
                        else None
                    )
                    guiding_self_condition_features = (
                        guiding_batch["self_condition_features"].detach()
                        if (
                            "self_condition_features" in guiding_batch
                            and guiding_batch["self_condition_features"] != None
                        )
                        else None
                    )

                    velocity = guiding_batch["out"] + cfg * (
                        main_batch["out"] - guiding_batch["out"]
                    )

                else:
                    velocity = main_batch["out"]

                if return_self_condition:
                    if "diff" in main_batch:
                        self_condition.append(
                            main_batch["diff"].detach().cpu()
                        )  # (b, C, h, w) — full channels kept for downstream PCA viz
                    else:
                        self_condition.append(
                            main_self_condition.detach().cpu()
                        )  # (b, C, h, w) — full channels kept for downstream PCA viz
                    fake_x_0 = x_t + (1 - t) * velocity
                    images.append(fake_x_0.to(dtype).detach().cpu())

                ## Updating the x_t using the Euler method ##
                x_t = x_t + step_size * velocity

        ## Returning the final image ##
        if return_self_condition:
            return images, self_condition
        return x_t.to(dtype), self_condition


class JiTHeunSampler:
    def __init__(
        self,
        in_resolution,
        in_channels,
        num_tokens=None,
        self_cond_in_dim=None,
        cfg_interval=[0.0, 1.0],
        t_eps=1e-5,
    ):
        """Constructor."""
        self.in_resolution = (
            [in_resolution, in_resolution]
            if isinstance(in_resolution, int)
            else in_resolution
        )
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.self_cond_in_dim = self_cond_in_dim
        self.cfg_interval = cfg_interval
        self.t_eps = t_eps

    @torch.no_grad()
    def _single_step(self, model, cond_batch, null_batch, t, cfg):
        """Performs a single step. This is very similar to a single Euler step."""

        cond_batch = model(cond_batch)
        t_now = cond_batch["timesteps"]

        low, high = self.cfg_interval

        if cfg > 1.0 and (t < high and ((low == 0) | (t > low))):

            null_batch = model(null_batch)
            null_x = null_batch["out"]
            cond_x = cond_batch["out"]
            velocity_null_t = (null_x - null_batch["noisy_imgs"]) / (
                1 - t_now.view(-1, 1, 1, 1)
            ).clamp(min=self.t_eps)
            velocity_cond_t = (cond_x - cond_batch["noisy_imgs"]) / (
                1 - t_now.view(-1, 1, 1, 1)
            ).clamp(min=self.t_eps)
            velocity = velocity_null_t + cfg * (velocity_cond_t - velocity_null_t)
        else:
            velocity = (cond_batch["out"] - cond_batch["noisy_imgs"]) / (
                1 - t_now.view(-1, 1, 1, 1)
            ).clamp(min=self.t_eps)

        return velocity, cond_batch, null_batch

    @torch.no_grad()
    def sample(
        self,
        model,
        null_conditions,
        conditions,
        null_condition_masks=None,
        condition_masks=None,
        num_sampling_steps=250,
        cfg=1.0,
        generator=None,
        x_t=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
        return_self_condition=False,
        **kwargs,
    ):
        batch_size = len(null_conditions)

        ## Setting the initial noise ##
        if x_t is None:
            x_t = torch.randn(  # Shape: (b, c, h, w)
                batch_size,
                self.in_channels,
                *self.in_resolution,
                generator=generator,
                device=device,
                dtype=dtype,
            )
        ## Setting the step size ##
        step_size = 1.0 / num_sampling_steps

        if self.self_cond_in_dim is not None:
            cond_self_condition = torch.zeros(
                batch_size,
                self.num_tokens,
                self.self_cond_in_dim,
                device=x_t.device,
                dtype=x_t.dtype,
            )
            null_self_condition = torch.zeros(
                batch_size,
                self.num_tokens,
                self.self_cond_in_dim,
                device=x_t.device,
                dtype=x_t.dtype,
            )

        else:
            cond_self_condition = None
            null_self_condition = None

        self_condition = [
            (
                cond_self_condition.detach().cpu()
                if cond_self_condition is not None
                else None
            )
        ]
        images = [x_t.to(dtype).detach().cpu()]

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        ):
            ## Looping over the sampling steps ##
            for step in tqdm(range(num_sampling_steps), total=num_sampling_steps):
                ## Getting the timesteps ##
                t = step / num_sampling_steps
                t_now = torch.tensor([t] * batch_size, device=device, dtype=dtype)
                t_next = (step + 1) / num_sampling_steps

                cond_batch = {
                    "noisy_imgs": x_t,
                    "conditions": conditions,
                    "condition_masks": condition_masks,
                    "timesteps": t_now,
                    "self_condition": cond_self_condition,
                }

                null_batch = {
                    "noisy_imgs": x_t,
                    "conditions": null_conditions,
                    "condition_masks": null_condition_masks,
                    "timesteps": t_now,
                    "self_condition": null_self_condition,
                    # "self_condition": cond_self_condition,
                }

                velocity_pred_t, cond_batch, null_batch = self._single_step(
                    model, cond_batch, null_batch, t, cfg
                )

                if return_self_condition:
                    if "diff" in cond_batch:
                        self_condition.append(
                            cond_batch["diff"].detach().cpu()
                        )  # (b, C, h, w) — full channels kept for downstream PCA viz
                    else:
                        current_self_condition = (
                            cond_self_condition.detach().cpu()
                            if cond_self_condition is not None
                            else None
                        )
                        self_condition.append(current_self_condition)
                    fake_x_0 = x_t + (1 - t) * velocity_pred_t
                    images.append(fake_x_0.to(dtype).detach().cpu())
                ## Last step always uses Euler ##
                if step == num_sampling_steps - 1:
                    x_t = x_t + step_size * velocity_pred_t

                ## Rest Heun ##
                else:
                    x_t_next = x_t + step_size * velocity_pred_t

                    ## Next half step ##
                    cond_batch["noisy_imgs"] = x_t_next
                    null_batch["noisy_imgs"] = x_t_next

                    cond_batch["timesteps"] = torch.tensor(
                        [t_next] * batch_size, device=device, dtype=dtype
                    )
                    null_batch["timesteps"] = torch.tensor(
                        [t_next] * batch_size, device=device, dtype=dtype
                    )
                    velocity_pred_t_next, cond_batch, null_batch = self._single_step(
                        model, cond_batch, null_batch, t_next, cfg
                    )

                    ## Updating the x_t using the Heun method ##
                    x_t = x_t + step_size * 0.5 * (
                        velocity_pred_t + velocity_pred_t_next
                    )

                    ## Updating the self_condition ##
                    cond_self_condition = (
                        cond_batch["self_condition"].detach()
                        if (
                            "self_condition" in cond_batch
                            and cond_batch["self_condition"] != None
                        )
                        else None
                    )

                    null_self_condition = (
                        null_batch["self_condition"].detach()
                        if (
                            "self_condition" in null_batch
                            and null_batch["self_condition"] != None
                        )
                        else None
                    )

        ## Returning the final image ##
        if return_self_condition:
            return images, self_condition
        return x_t.to(dtype), self_condition