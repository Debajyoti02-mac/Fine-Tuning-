                PRETRAINED MODEL
                
                 DistilGPT-2
                 82M parameters
                       │
                       ▼
                4-bit quantization
                       │
                       ▼
                 Freeze base model
                       │
                       ▼
              Add LoRA matrices
                  A       B
                  │       │
                  └───┬───┘
                      ▼
                Train only A + B
                      │
                      ▼
                 LoRA adapter
                      │
                      ▼
               Adapted behavior
