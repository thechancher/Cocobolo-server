# App Server of ViT of the Cocobolo Project

ViT for the identification of tree species in Costa Rica

This server app runs the ViT model and the web app

## Web app

### In 'template' folder

copy the 'index.html' file

change the href
``` html
<base href="./">
```
to
``` html
<base href="./../static/">
```

### In 'static' folder

copy all the rest files from the web app project as *.js, *.css, asset/, svg/, etc.

### In 'models' folder

copy the trained model .pth to use to evaluate the image

### In 'images' folder

the server application keeps image under evaluation

## Service app

### Install the service

copy the 'ViT.service' to '/etc/systemd/system'

### To start the service

``` bash
sudo systemctl start ViT.service
```

### To restart the service

``` bash
sudo systemctl restart ViT.service
```

### To stop the service
``` bash
sudo systemctl stop ViT.service
```


### To check the status service

``` bash
sudo systemctl status ViT.service
```

### To see the services outputs

``` bash
sudo journalctl -u ViT.service
```