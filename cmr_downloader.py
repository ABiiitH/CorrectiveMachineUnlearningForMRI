# script for installing the poisoned dataset (CMRxRecon)
# use an authtoken or credentials with the syn.login 

import synapseclient
import sys

def main():
    syn_id = "syn53162168"  
    syn = synapseclient.Synapse(cache_root_dir=".")
    
    try:
        syn.login()
        print(f"Downloading Synapse file {syn_id}...")
        file_entity = syn.get(syn_id, downloadLocation=".")
        print(f"Downloaded file to: {file_entity.path}")
        print(file_entity)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
