# Copyright 2011 James McCauley
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



"""
./pox.py samples.pretty_log openflow.discovery misc.gephi_topo host_tracker forwarding.l2_learning
"""

def launch ():
    import pox.samples.pretty_log
    pox.samples.pretty_log.launch()
    import pox.openflow.keepalive
    pox.openflow.keepalive.launch()
    import pox.openflow.discovery
    pox.openflow.discovery.launch()
    import pox.misc.gephi_topo
    pox.misc.gephi_topo.launch()
    import pox.openflow.topology
    pox.openflow.topology.launch()
    import pox.host_tracker
    pox.host_tracker.launch()
    #import pox.openflow.spanning_tree
    #pox.openflow.spanning_tree.launch()
    import pox.forwarding.l2_learning
    pox.forwarding.l2_learning.launch()
  
    
    #./pox.py log.level --DEBUG messenger messenger.tcp_transport messenger.example
   
    import pox.messenger  
    pox.messenger.launch()
    import pox.messenger.tcp_transport
    pox.messenger.tcp_transport.launch()
    import pox.messenger.example
    pox.messenger.example.launch()
   
