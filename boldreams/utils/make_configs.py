import json

bb_names=['alexnet',
          'vgg11',
          'RN50_clip_add_last']

bb_trains=['True','False']


max_percents= ['1','5','10','15','20','25','50','75','100']

SYSTEM='cedar'

UPTO=16
for bb_name in bb_names:
    for bb_train in bb_trains:
        for max_percent in max_percents:
            dic={"SYSTEM":SYSTEM,
                 "backbone_name":bb_name,
                 "UPTO":UPTO,
                 "epochs":10,
                 "train_backbone":bb_train,
                 "max_filters":"False",
                 "max_percent":int(max_percent)}
            #print(dic)


            json_file_path = "bb-%s_upto-%d_bbtrain-%s_max-channels_%d.json"%(bb_name,UPTO,bb_train,int(max_percent))

            # Save the dictionary as a JSON file
            #with open(json_file_path, 'w') as json_file:
            #    json.dump(dic, json_file)

            print(f"The dictionary has been saved as {json_file_path}")