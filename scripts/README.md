# scripts

Supporting code and build files for developing and operating environments.

Keep environment-specific files in `scripts/<env-name>/`. This includes sandbox image definitions and build helpers, setup utilities, and other development code that is not part of the installable environment package. Synthetic data generation and validation code remains in `generators/<env-name>/`.

Repository-wide utilities, such as `install.sh`, may remain directly under `scripts/`.
