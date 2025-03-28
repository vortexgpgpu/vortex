# Apptainer Build Process

Use the Slurm scheduler to request an interactive job on Flubber9
```
salloc -p rg-fpga --nodes=1 --ntasks-per-node=64 --mem=16G --nodelist flubber9 --time=01:00:00
```

Go to `apptainer` directory

```
$ pwd
vortex/miscs/apptainer

$  apptainer build --no-https vortex_fpga.sif vortex_fpga.def 

```
To start the apptainer,
```
$ apptainer shell --fakeroot  --cleanenv --writable-tmpfs --bind /opt/xilinx/:/opt/xilinx/ --bind /netscratch/rn84/devnull:/dev/null --bind /dev/bus/usb,/sys/bus/pci --bind /projects:/projects --bind /tools:/tools  --bind /netscratch:/netscratch vortex_fpga.sif
```

Inside the Apptainer,
```
Apptainer> lsusb

should show devices connected to machine on which you are running this command
```


```
Apptainer> source /opt/xilinx/xrt/setup.sh
Apptainer> source /tools/reconfig/xilinx/Vitis/2023.1/settings64.sh

Apptainer> platforminfo -l
```




