import json
import pandas as pd

"""
TODO:
Getters and setters for token mapping
Given cols and assoc. values in DF, return token IDs
Given DF, make mapping and encode
Given rule outputs, find matching samples
Given rule outputs, return cols and assoc. values
"""

class TokenTranslator:
    """Hashing-based technique to reduce sequence datasets in CSV format 
    to the lexical format used by SPMF (https://www.philippe-fournier-viger.com/spmf). 
    Useful for applying Sequential Pattern/Rule Mining approaches to non-text data."""
    def __init__(self, data:pd.DataFrame, mapping:str|dict=None) -> None:
        self.data = data
        self.set_mapping(mapping)
                
    def set_mapping(self, mapping:str|dict) -> dict:
        self.mapping = mapping
        if type(mapping) == str:
            # File path passed
            with open(mapping, 'r') as map_fp:
                self.mapping = json.load(map_fp)
        return self.mapping
    
    def get_mapping(self) -> dict|None:
        return self.mapping
    
    def build_mapping(self, set=True, save_path:str=None) -> dict:
        token_dict = {}
        i = 0
        for sample, row in self.data.iterrows():
            for index, value in row.items():
                key = self.make_key(index, value)
                if key not in token_dict:
                    token_dict[key] = str(i)
                    i += 1
        if set == True: self.mapping = token_dict

        if save_path is not None:
            with open(save_path, 'w') as fp:
                json.dump(token_dict, fp)

        return token_dict
    
    def check_mapping(self) -> None:
        assert type(self.mapping) == dict, "Attribute ```mapping``` must be of type dict! Make sure it is set with ```set_mapping()```"

    def encode_data(self, data:pd.DataFrame=None, save_path:str=None) -> str:
        self.check_mapping()

        if data is None: data = self.data
        output = ""
        for sample, row in data.iterrows():
            for index, value in row.items():
                key = self.make_key(index, value)
                output += self.mapping[key] + " -1 "
            output += "-2\n"
        
        if save_path is not None: 
            with open(save_path, 'w') as fp:
                fp.write(output)

        return output
    
    def get_token(self, col_name:str, value) -> str:
        self.check_mapping()
        return self.mapping[self.make_key(col_name, value)]

    def make_key(self, col_name:str, value) -> str:
        return col_name + "@" + str(value)
    
    def get_key(self, token_id:str) -> str:
        self.check_mapping()
        key_list = list(self.mapping.keys())
        try: 
            key = key_list[int(token_id)]
        except: 
            assert "token_id is not in mapping!"
        return key
    
    def get_column_value(self, token_id:str) -> list:
        key = self.get_key(token_id)
        return key.split('@')
    
    def get_column(self, token_id:str) -> str:
        return self.get_column_value(token_id)[0]
    
    def get_value(self, token_id:str) -> str:
        return self.get_column_value(token_id)[1]