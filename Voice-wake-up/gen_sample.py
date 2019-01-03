from aip import AipSpeech
import os

class Sample:

    def __init__(self):

        self.count=0
        self.baidu_server = "https://openapi.baidu.com/oauth/2.0/token?"  # 认证的url
        self.grant_type = "client_credentials"

        self.app_id="tgj891"
        self.api_key = "UUx3lDmj7pdxW8jpTbHsmRHE"  # 填写API Key
        self.secret_key = "bmiG20py26RiGvCPF62T1G4EDAmIpD65"  # 填写Secret Key


        self.aipSpeech=AipSpeech(self.app_id,self.api_key,self.secret_key)


    def sampling(self,istrue,text,path):

        if istrue:
            self.vol = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 音量，取值0-15，默认为5中音量
            self.per = [0, 1, 3, 4]  # 发音人选择, 0为女声，1为男声，3为情感合成-度逍遥，4为情感合成-度丫丫
            self.spd = [4, 5, 6, 7, 8, 9]  # 语速，取值0-9，默认为5中语速
            self.pit = [4, 5, 6, 7, 8, 9]  # 音调，取值0-9，默认为5中语调

        else:
            self.vol = [5]  # 音量，取值0-15，默认为5中音量
            self.per = [0, 1]  # 发音人选择, 0为女声，1为男声，3为情感合成-度逍遥，4为情感合成-度丫丫
            self.spd = [5]  # 语速，取值0-9，默认为5中语速
            self.pit = [5]  # 音调，取值0-9，默认为5中语调

        for vol in self.vol:
            for per in self.per:
                for spd in self.spd:
                    for pit in self.pit:
                        options={"vol":vol,"per":per,"spd":spd,"pit":pit}

                        try:
                            result=self.aipSpeech.synthesis(text,"zh",1,options)
                        except:#连接异常
                            continue

                        if not isinstance(result,dict):
                            if istrue:#正样本
                                self.mp3file=os.path.join(path,"1-"+str(self.count)+".mp3")
                            else:#负样本
                                self.mp3file=os.path.join(path,"0-"+str(self.count)+".mp3")

                            with open(self.mp3file,"wb") as f:
                                f.write(result)
                            self.count+=1

if __name__ == '__main__':
    sample=Sample()
    #正样本
    # sample.sampling(True,"小白小白",r"D:\语音\音频")

    #负样本
    # strs=open(r"D:\语音\标签\yangben.txt").readlines()
    # for data in strs:
    #     data=data.strip()
    #     print(data)
    #     sample.sampling(False,"小"+data+"小"+data,r"D:\语音\音频")

    #产生test样本
    sample.sampling(False, "柠睿柠睿", r"D:\语音\柠睿柠睿")