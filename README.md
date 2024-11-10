# bells_ml
Bell model, incorporating semi-realistic pendulum physics.
Designed to be used to train a neural network to ring, as a nice 1D problem.

#Some instructions

To get the interactive bell to work (no ml stuff so far):
This is run using main.py in the bells_ml folder

To edit:

Load anaconda prompt, the environment and navigate to the correct folder. Get updates from GitHub in case

```
activate bells
cd \Users\eleph\OneDrive\Documents\bells\bells_ml
git pull
```

For a test run, in main.py ensure the correct asyncio is applied:
```
import nest_asyncio
nest_asyncio.apply()
```
(Remove these to make it work in the browser, I'm pretty sure). 

This will open the pygame window and you can play with it.

To compile for exporting to the browser:

Remove the lines of code about nest_asyncio (above) then navigate to the folder above and run pygbag
```
pip install pygbag
cd \Users\eleph\OneDrive\Documents\bells\
pygbag bells_ml
```
Then navigate to 
```
localhost:8000
```
to test.

To host it publicly, copy the .html and .apk files into the public_html folder. Metadata etc. can be changed by having a fiddle around with the .html






