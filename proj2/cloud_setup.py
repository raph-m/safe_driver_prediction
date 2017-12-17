import subprocess

"""
sudo apt update
sudo apt install python python-dev python3 python3-dev
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

"""

bashCommand = "cwm --rdf test.rdf --ntriples > test.nt"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()