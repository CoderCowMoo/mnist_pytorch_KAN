# KAN MNIST

Similar to my other repo, but using a KAN instead of MLP, with as much of the same parameters as possible.

References:
- https://paperswithcode.com/paper/kan-kolmogorov-arnold-networks
- https://github.com/KindXiaoming/pykan/
- https://arxiv.org/pdf/2404.19756v1

Trouble:
Currently having some trouble right now, with torch.reshape in the train cell.

```python
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
Cell In[12], line 20
     17 def test_acc():
     18     return torch.mean((torch.round(model(dataset['test_input'])[:,0]) == dataset['test_label']).float())
---> 20 results = model.train(dataset, opt="LBFGS", steps=20, metrics=(train_acc, test_acc))
     21 results['train_acc'][-1], results['test_acc'][-1]

File c:\Users\Moo\AppData\Local\Programs\Python\Python311\Lib\site-packages\kan\KAN.py:913, in KAN.train(self, dataset, opt, steps, log, lamb, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff, update_grid, grid_update_num, loss_fn, lr, stop_grid_update_step, batch, small_mag_threshold, small_reg_factor, metrics, sglr_avoid, save_fig, in_vars, out_vars, beta, save_fig_freq, img_folder, device)
    910 test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)
    912 if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid:
--> 913     self.update_grid_from_samples(dataset['train_input'][train_id].to(device))
    916 if opt == "LBFGS":
    917     optimizer.step(closure)

File c:\Users\Moo\AppData\Local\Programs\Python\Python311\Lib\site-packages\kan\KAN.py:242, in KAN.update_grid_from_samples(self, x)
    219 '''
    220 update grid from samples
    221 
   (...)
    239 tensor([0.0128, 1.0064, 2.0000, 2.9937, 3.9873, 4.9809])
    240 '''
    241 for l in range(self.depth):
--> 242     self.forward(x)
    243     self.act_fun[l].update_grid_from_samples(self.acts[l])
...
     75 if func in _device_constructors() and kwargs.get('device') is None:
     76     kwargs['device'] = self.device
---> 77 return func(*args, **kwargs)

```