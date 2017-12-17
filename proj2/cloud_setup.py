import subprocess

"""
sudo apt update
sudo apt install python python-dev python3 python3-dev
sudo apt-get install python3-setuptools
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
sudo pip install --upgrade virtualenv
sudo pip install virtualenvwrapper
echo "export WORKON_HOME=$HOME/.virtualenvs" >> .bashrc
echo "export PROJECT_HOME=$HOME/Devel" >> .bashrc
echo "source /usr/local/bin/virtualenvwrapper.sh" >> .bashrc
echo "source "/usr/bin/virtualenvwrapper.sh"" >> .bashrc
echo "export WORKON_HOME="/opt/virtual_env/"" >> .bashrc
source `which virtualenvwrapper.sh`
sudo pip install pandas
sudo pip install requests
git clone https://github.com/raph-m/safe_driver_prediction
cd safe_driver_prediction/proj2
python gdrive.py 1EQ0zE_2WLQdNIepWUjroPyGmi-dvN5KK ../../data.zip
cd ..
cd ..
sudo apt-get install unzip
unzip data.zip
cd safe_driver_prediction
git pull origin master
python proj2/feature_engineering.py train ../../data/ 3000000
"""

bashCommand = "cwm --rdf test.rdf --ntriples > test.nt"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()