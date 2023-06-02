# GAN Training Script

This script is used for training a Generative Adversarial Network (GAN) consisting of a generator, a discriminator, and a classifier. The script is written in PyTorch.

## Setup

```python
# Network Setup
generator = Generator(args.db, args.z_dim, args.cc_dim, args.dc_dim)
discriminator = Discriminator(args.db, args.featu_dim)
classifier = Classifier(args.db, args.cc_dim, args.dc_dim)

# Optimizers Setup
g_optimizer = optim.Adam(generator.parameters(), args.lrG, [args.beta1, args.beta2])
d_optimizer = optim.Adam(discriminator.parameters(), args.lrD, [args.beta1, args.beta2])
c_optimizer = optim.Adam(classifier.parameters(), args.lrD, [args.beta1, args.beta2])

# CUDA Support
The script supports CUDA if available. The following line is used to move the networks to GPU:

if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
    classifier.cuda()
    
# Sample Generation
After the GAN is trained, you can generate samples using the following code:
if (i + 1) % args.sample_step == 0:
    # Generate samples
    fake_images = generator(a)
    torchvision.utils.save_image(fake_images.data,
                                    os.path.join(args.sample_path, #args.sample_path
                                        'generated-%d-%d.png' % (epoch + 1, i + 1)), nrow=10)#列数
