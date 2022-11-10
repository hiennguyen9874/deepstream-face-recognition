- https://forums.developer.nvidia.com/t/how-to-create-opencv-gpumat-from-nvstream/70680/18
- Flip video
- Mount opencv from host into docker using nvidia-container-runtime: https://github.com/dusty-nv/jetson-containers/issues/5
- TensorRT custom error when export onnx to tensorrt on jetson
    ```
    [11/05/2022-02:13:31] [W] [TRT] Skipping tactic 3 due to insuficient memory on requested size of 2243 detected for tactic 4.
    Try decreasing the workspace size with IBuilderConfig::setMaxWorkspaceSize().
    [11/05/2022-02:13:32] [W] [TRT] Tactic Device request: 2218MB Available: 2189MB. Device memory is insufficient to use tactic.
    [11/05/2022-02:13:32] [W] [TRT] Skipping tactic 3 due to insuficient memory on requested size of 2218 detected for tactic 4.
    Try decreasing the workspace size with IBuilderConfig::setMaxWorkspaceSize().
    [11/05/2022-02:14:30] [W] [TRT] Tactic Device request: 4461MB Available: 2343MB. Device memory is insufficient to use tactic.
    [11/05/2022-02:14:30] [W] [TRT] Skipping tactic 3 due to insuficient memory on requested size of 4461 detected for tactic 4.
    Try decreasing the workspace size with IBuilderConfig::setMaxWorkspaceSize().
    [11/05/2022-02:14:31] [W] [TRT] Tactic Device request: 4423MB Available: 2343MB. Device memory is insufficient to use tactic.
    [11/05/2022-02:14:31] [W] [TRT] Skipping tactic 3 due to insuficient memory on requested size of 4423 detected for tactic 4.
    Try decreasing the workspace size with IBuilderConfig::setMaxWorkspaceSize().
    [11/05/2022-02:14:49] [W] [TRT] Tactic Device request: 4327MB Available: 2386MB. Device memory is insufficient to use tactic.
    [11/05/2022-02:14:49] [W] [TRT] Skipping tactic 3 due to insuficient memory on requested size of 4327 detected for tactic 4.
    Try decreasing the workspace size with IBuilderConfig::setMaxWorkspaceSize().
    [11/05/2022-02:14:50] [W] [TRT] Tactic Device request: 4308MB Available: 2387MB. Device memory is insufficient to use tactic.
    [11/05/2022-02:14:50] [W] [TRT] Skipping tactic 3 due to insuficient memory on requested size of 4308 detected for tactic 4.
    Try decreasing the workspace size with IBuilderConfig::setMaxWorkspaceSize().
    [11/05/2022-02:15:13] [W] [TRT] Tactic Device request: 4309MB Available: 2389MB. Device memory is insufficient to use tactic.
    [11/05/2022-02:15:13] [W] [TRT] Skipping tactic 3 due to insuficient memory on requested size of 4309 detected for tactic 4.
    Try decreasing the workspace size with IBuilderConfig::setMaxWorkspaceSize().
    [11/05/2022-02:15:14] [W] [TRT] Tactic Device request: 4299MB Available: 2388MB. Device memory is insufficient to use tactic.
    [11/05/2022-02:15:14] [W] [TRT] Skipping tactic 3 due to insuficient memory on requested size of 4299 detected for tactic 4.
    Try decreasing the workspace size with IBuilderConfig::setMaxWorkspaceSize().
    ```
