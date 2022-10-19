def preprocess(element):
    """
    Function that takes an element as a json file, and transforms it into an array
    """
    #DICTIONARY OF ZIP CODE VALUES 
    zip_code_dict_xx = {
        'be_zip_10': 1.53,
        'be_zip_11': 1.68,
        'be_zip_12': 1.66,
        'be_zip_13': 1.29,
        'be_zip_14': 1.18,
        'be_zip_15': 1.24,
        'be_zip_16': 1.31,
        'be_zip_17': 1.23,
        'be_zip_18': 1.22,
        'be_zip_19': 1.5,
        'be_zip_20': 1.53,
        'be_zip_21': 1.17,
        'be_zip_22': 1.13,
        'be_zip_23': 1.12,
        'be_zip_24': 1.03,
        'be_zip_25': 1.24,
        'be_zip_26': 1.27,
        'be_zip_27': 1.11, 
        'be_zip_28': 1.22,
        'be_zip_29': 1.3,
        'be_zip_30': 1.58,
        'be_zip_31': 1.18,
        'be_zip_32': 1.1,
        'be_zip_33': 1.07,
        'be_zip_34': 0.87,
        'be_zip_35': 1.13,
        'be_zip_36': 1.0,
        'be_zip_37': 0.9,
        'be_zip_38': 0.94,
        'be_zip_39': 1.0,
        'be_zip_40': 0.93,
        'be_zip_41': 0.85,
        'be_zip_42': 0.86,
        'be_zip_43': 0.87,
        'be_zip_44': 0.81,
        'be_zip_45': 0.76,
        'be_zip_46': 0.95,
        'be_zip_47': 0.98,
        'be_zip_48': 0.85,
        'be_zip_49': 0.94,
        'be_zip_50': 0.97,
        'be_zip_51': 1.0,
        'be_zip_52': 0.77,  
        'be_zip_53': 0.87,
        'be_zip_54': 0.77,
        'be_zip_55': 0.76,
        'be_zip_56': 0.67,
        'be_zip_57': 0.77,
        'be_zip_58': 0.77,
        'be_zip_59': 0.77,
        'be_zip_60': 0.64,
        'be_zip_61': 0.74,
        'be_zip_62': 0.78,
        'be_zip_63': 0.69,
        'be_zip_64': 0.66,
        'be_zip_65': 0.67,
        'be_zip_66': 0.91,
        'be_zip_67': 0.97,
        'be_zip_68': 0.84,
        'be_zip_69': 0.83,
        'be_zip_70': 0.8,
        'be_zip_71': 0.69,
        'be_zip_72': 0.67,
        'be_zip_73': 0.58,
        'be_zip_75': 0.86,
        'be_zip_76': 0.66,
        'be_zip_77': 0.79,
        'be_zip_78': 0.91,
        'be_zip_79': 0.66,
        'be_zip_80': 1.34,
        'be_zip_81': 1.25,
        'be_zip_82': 1.32,
        'be_zip_83': 2.12,
        'be_zip_84': 1.43,
        'be_zip_85': 1.06,
        'be_zip_86': 1.61,
        'be_zip_87': 1.16,
        'be_zip_88': 0.98,
        'be_zip_89': 0.95,
        'be_zip_90': 1.46,
        'be_zip_91': 1.13,
        'be_zip_92': 1.11,
        'be_zip_93': 1.03,
        'be_zip_94': 1.0,
        'be_zip_95': 0.96,
        'be_zip_96': 0.94,
        'be_zip_97': 1.11,
        'be_zip_98': 1.27,
        'be_zip_99': 1.16
        }
    
    #DICTIONARY OF STATE OF BUILDING
    state_of_the_building_dict = {
        "NEW": 1.0,
        "GOOD": 0.79285,
        "TO RENOVATE": 0.56664,
        "JUST RENOVATED": 0.93115,
        "TO REBUILD": 0.46920
        }
    
    #TRANSFORM DATA
    transformed_elment = {
        'number_of_bedrooms': element.data.rooms_number,
        'surface': element.data.area,    
        'fully_equipped_kitchen': 1 if element.data.equipped_kitchen == True else 0,
        'open_fire': 1 if element.data.open_fire == True else 0,
        'terrace_surface': element.data.terrace_area if element.data.terrace_area is not None else 0,
        'garden': 1 if element.data.garden == True else 0,
        'number_of_facades': element.data.facades_number if element.data.facades_number is not None else 1 if element.data.property_type == 'APARTMENT' else 2,
        'swimming_pool': 1 if element.data.swimming_pool == True else 0,
        'state_of_the_building': state_of_the_building_dict[element.data.building_state] if element.data.building_state is not None else 0.87252,
        'zip_code_ratio': zip_code_dict_xx['be_zip_'+str(element.data.zip_code)[:2]],
        'HOUSE': 1 if element.data.property_type == 'HOUSE' else 0,
        'APARTMENT': 1 if element.data.property_type == 'APARTMENT' else 0
        }
    return list(transformed_elment.values())