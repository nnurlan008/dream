<rspec xmlns="http://www.geni.net/resources/rspec/3" xmlns:emulab="http://www.protogeni.net/resources/rspec/ext/emulab/1" xmlns:tour="http://www.protogeni.net/resources/rspec/ext/apt-tour/1" xmlns:jacks="http://www.protogeni.net/resources/rspec/ext/jacks/1" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.geni.net/resources/rspec/3    http://www.geni.net/resources/rspec/3/request.xsd" type="request"><rspec_tour xmlns="http://www.protogeni.net/resources/rspec/ext/apt-tour/1"><description xmlns="" type="markdown">RDMA connection</description></rspec_tour>
<node client_id="bf1" exclusive="true">
    <sliver_type name="raw">
      <disk_image name="urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU22-64-STD"/>
    </sliver_type>
    <hardware_type name="r7525"/>
    <interface client_id="bf1:interface-0"/>
    <interface client_id="bf1:interface-2"/>
    <interface client_id="bf1:interface-4"/>
  <services xmlns="http://www.geni.net/resources/rspec/3"/></node><node client_id="bf2" exclusive="true">
    <sliver_type name="raw">
      <disk_image name="urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU22-64-STD"/>
    </sliver_type>
    <hardware_type name="r7525"/>
    <interface client_id="bf2:interface-1"/>
    <interface client_id="bf2:interface-3"/>
    <interface client_id="bf2:interface-5"/>
  <services xmlns="http://www.geni.net/resources/rspec/3"/></node><link client_id="link-0">
    <interface_ref client_id="bf2:interface-1"/>
    <interface_ref client_id="bf1:interface-0"/>
    
    
    <ns0:site xmlns:ns0="http://www.protogeni.net/resources/rspec/ext/jacks/1" id="undefined"/>
  <property xmlns="http://www.geni.net/resources/rspec/3" source_id="bf2:interface-1" dest_id="bf1:interface-0" capacity="40000000"/><property xmlns="http://www.geni.net/resources/rspec/3" source_id="bf1:interface-0" dest_id="bf2:interface-1" capacity="40000000"/></link><link client_id="link-1">
    <interface_ref client_id="bf1:interface-2"/>
    <interface_ref client_id="bf2:interface-3"/>
    
    
    <ns1:site xmlns:ns1="http://www.protogeni.net/resources/rspec/ext/jacks/1" id="undefined"/>
  <property xmlns="http://www.geni.net/resources/rspec/3" source_id="bf1:interface-2" dest_id="bf2:interface-3" capacity="40000000"/><property xmlns="http://www.geni.net/resources/rspec/3" source_id="bf2:interface-3" dest_id="bf1:interface-2" capacity="40000000"/></link><link client_id="link-2">
    <interface_ref client_id="bf1:interface-4"/>
    <interface_ref client_id="bf2:interface-5"/>
    
    
    <ns2:site xmlns:ns2="http://www.protogeni.net/resources/rspec/ext/jacks/1" id="undefined"/>
  <property xmlns="http://www.geni.net/resources/rspec/3" source_id="bf1:interface-4" dest_id="bf2:interface-5" capacity="1000000"/><property xmlns="http://www.geni.net/resources/rspec/3" source_id="bf2:interface-5" dest_id="bf1:interface-4" capacity="1000000"/></link></rspec>
