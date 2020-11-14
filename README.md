# ADI

Install POX controller https://github.com/noxrepo/pox

Install mininet

Copy the file startl2.py in pox/ext folder

Replace the files in folder messenger in pox/pox with the files inside pox-messeenger folder.

Run pox as ./pox.py startl2

Run mininet as sudo mn --topo linear,4 --controller remote

In mininet run s1 xterm, it will open another terminal for you. download the test_s1.py file and run it in this xterm terminal as python3 test_s1.py.
