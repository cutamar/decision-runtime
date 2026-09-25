# Example data

`labeled-synthetic.csv` contains four invented English routing examples written
for this repository. It has no customer records and is covered by the runtime
Apache-2.0 license. It demonstrates the local evaluation input format; four
rows cannot establish useful model quality.

`router_integration.py` shows how a customer application can use an exported
bundle and send abstentions to manual review. A signed bundle requires the
independently trusted PEM public key. The example does not execute any external
routing action.
