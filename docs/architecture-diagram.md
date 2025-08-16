# DiffSynth-Engine Architecture Diagram

This mermaid diagram shows the overall architecture and flow of the DiffSynth-Engine, which is a high-performance inference engine for diffusion models.

```mermaid
graph TB
    %% Input Layer
    A[User Input: Prompt, Image, Parameters] --> B[Configuration]
    B --> B1[Pipeline Config<br/>FluxPipelineConfig/SDXLPipelineConfig/etc.]
    
    %% Model Fetching & Loading
    B1 --> C[Model Fetching System]
    C --> C1[fetch_model<br/>HuggingFace/CivitAI/ModelScope]
    C1 --> C2[State Dict Loading<br/>SafeTensors/GGUF]
    C2 --> C3[Model Conversion<br/>Diffusers → DiffSynth Format]
    
    %% Pipeline Factory
    C3 --> D{Pipeline Type}
    D -->|Text-to-Image| E[FluxImagePipeline]
    D -->|SDXL| F[SDXLImagePipeline]
    D -->|SD| G[SDImagePipeline]
    D -->|Video| H[WanVideoPipeline]
    D -->|Qwen Image| I[QwenImagePipeline]
    
    %% Main Pipeline Flow (using Flux as example)
    E --> J[Model Initialization]
    
    %% Text Processing
    J --> K[Text Processing]
    K --> K1[CLIPTokenizer + T5TokenizerFast]
    K1 --> K2[FluxTextEncoder1 + FluxTextEncoder2]
    K2 --> K3[Text Embeddings]
    
    %% Image Processing (if img2img)
    J --> L[Image Processing]
    L --> L1[Image Preprocessing]
    L1 --> L2[FluxVAEEncoder]
    L2 --> L3[Latent Space]
    
    %% Noise & Sampling Setup
    J --> M[Noise & Sampling Setup]
    M --> M1[Noise Generation + Dynamic Shifting]
    M --> M2[RecifitedFlowScheduler → Timesteps]
    M --> M3[FlowMatchEulerSampler → Strategy]
    
    %% Core Denoising Loop
    K3 --> N[Core Denoising Loop]
    L3 --> N
    M1 --> N
    M2 --> N
    M3 --> N
    
    N --> N1[FluxDiT Transformer<br/>+ ControlNet/IP-Adapter]
    N1 --> N2[Noise Prediction]
    N2 --> N3[Sampler Step]
    N3 --> N4{More Steps?}
    N4 -->|Yes| N1
    N4 -->|No| O[Final Latents]
    
    %% Image Decoding
    O --> P[FluxVAEDecoder]
    P --> Q[Generated Image]
    
    %% Performance Optimizations
    R[Performance Features]
    R --> R1[Memory Management<br/>CPU/GPU Offloading<br/>Sequential Offloading]
    R --> R2[Parallel Processing<br/>Tensor/Sequence Parallel<br/>CFG Parallel]
    R --> R3[Quantization<br/>FP8/GGUF Support<br/>Model Compilation]
    
    R1 --> J
    R2 --> J
    R3 --> J
    
    %% Model Customization
    S[Model Customization]
    S --> S1[LoRA Support<br/>Fused/Unfused Loading]
    S --> S2[Conditioning<br/>IP-Adapter/Redux]
    S --> S3[Control<br/>ControlNet/Inpainting]
    
    S1 --> N1
    S2 --> N1
    S3 --> N1
    
    %% Tools & Extensions
    T[Tools & Extensions]
    T --> T1[FluxInpaintingTool]
    T --> T2[FluxOutpaintingTool]
    T --> T3[FluxReferenceTools]
    T --> T4[FluxReplaceTool]
    
    T --> E
    
    %% Algorithm Foundation
    U[Algorithm Foundation]
    U --> U1[Noise Schedulers<br/>Beta/DDIM/Exponential/Karras]
    U --> U2[Samplers<br/>Euler/DPM++/DDPM/FlowMatch]
    
    U1 --> M
    U2 --> M
    
    style A fill:#e1f5fe
    style Q fill:#c8e6c9
    style N1 fill:#fff3e0
    style E fill:#f3e5f5
    style C fill:#fce4ec
    style R fill:#e8f5e8
    style S fill:#fff8e1
```

## Architecture Overview

The DiffSynth-Engine follows a modular architecture with these key components:

### 1. **Pipeline Layer** 
- **FluxImagePipeline**: Primary image generation pipeline using Flux models
- **SDXLImagePipeline**: Stable Diffusion XL pipeline
- **SDImagePipeline**: Standard Stable Diffusion pipeline  
- **WanVideoPipeline**: Video generation pipeline
- **QwenImagePipeline**: Qwen image generation pipeline

### 2. **Text Processing**
- **Tokenizers**: CLIPTokenizer and T5TokenizerFast for text preprocessing
- **Text Encoders**: CLIP and T5 models for text embedding generation
- **Prompt Encoding**: Converts text prompts to numerical embeddings

### 3. **Image Processing** 
- **VAE Encoder**: Encodes images to latent space representation
- **VAE Decoder**: Decodes latents back to pixel space
- **Preprocessing**: Image normalization and format conversion

### 4. **Noise Scheduling & Sampling**
- **Schedulers**: Define noise schedules (Beta, DDIM, Exponential, etc.)
- **Samplers**: Implement sampling strategies (Euler, DPM++, DDPM, etc.)
- **Timestep Management**: Controls the denoising process progression

### 5. **Core Denoising**
- **DiT (Diffusion Transformer)**: Main neural network for noise prediction
- **Attention Mechanisms**: Self-attention and cross-attention layers
- **ControlNet Integration**: Optional conditioning for guided generation

### 6. **Advanced Features**
- **LoRA Support**: Low-rank adaptation for model customization
- **IP-Adapter & Redux**: Image-based conditioning
- **Parallel Processing**: Multi-GPU and distributed inference
- **Memory Management**: CPU/GPU offloading and optimization
- **Quantization**: FP8 and other precision optimizations

### 7. **Model Management**
- **State Dict Handling**: Loading and converting model weights
- **Device Management**: GPU/CPU memory allocation
- **Model Lifecycle**: Loading, offloading, and cleanup

The engine supports multiple diffusion model formats (Flux, SD, SDXL, Wan, Qwen) while providing a unified interface and extensive optimization features for high-performance inference.