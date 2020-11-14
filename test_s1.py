import socket 
import threading 
import json 
import time


class JSONDestreamer (object): 
  import json 
  decoder = json.JSONDecoder() 
  def __init__ (self, callback = None): 
    self._buf = '' 
    self.callback = callback if callback else self.rx 
 
  def push (self, data): 
    if len(self._buf) == 0: 
      data = data.lstrip() 
    self._buf += data 
    try: 
      while len(self._buf) > 0: 
        r,off = self.decoder.raw_decode(self._buf) 
 
        self._buf = self._buf[off:].lstrip() 
        self.callback(r) 
    except ValueError: 
      pass 
 
  def rx (self, data): 
    import json 
    print "Recv:", json.dumps(data, indent=4) 
 
jd = JSONDestreamer() 
done = False 
 
def reader (socket): 
  global done 
  while True: 
    d = socket.recv(1024) 
    if d == "": 
      done = True 
      break 
    jd.push(d) 
 
cur_chan = None 
def channel (ch): 
  global cur_chan 
  cur_chan = ch 
 
import readline 
 
def main (addr = "127.0.0.1", port = 7790): 

  millis = int(round(time.time() * 1000))
  print millis

  print "Connecting to %s:%i" % (addr,port) 
  port = int(port) 
 
  sock = socket.create_connection((addr, port)) 
 
  t = threading.Thread(target=reader, args=(sock,)) 
  t.daemon = True 
  t.start() 
 
  msg="permission:yes,merit:low,affordability:some,lifetime:long,Update:yes,latency tolerance:high,trust:high,redundancy:yes" 
  while not done: 
    try: 
      #print ">", 
      m = "{\"CHANNEL\":\"upper\",\"msg\":"+"\""+msg+"\"}" 
      print (msg)	
      if len(m) == 0: continue 
      m = eval(m) 
      if not isinstance(m, dict): 
        continue 
      if cur_chan is not None and 'CHANNEL' not in m: 
        m['CHANNEL'] = cur_chan 
      m = json.dumps(m) 
      sock.send(m) 
      print('msg sent')	
      break; 
    except EOFError: 
      break 
    except KeyboardInterrupt: 
      break 
    except: 
      import traceback 
      traceback.print_exc() 
 
if __name__ == "__main__": 
  import sys 
  main(*sys.argv[1:]) 
else: 
  # This will get run if you try to run this as a POX component. 
  def launch (): 
    from pox.core import core 
    log = core.getLogger() 
    log.critical("This isn't a POX component.") 
    log.critical("Please see the documentation.") 
    raise RuntimeError("This isn't a POX component.") 
