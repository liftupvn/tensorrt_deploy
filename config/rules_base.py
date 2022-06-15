
import copy
import numpy as np

class ruleBase:
    def __init__(self, model_type = "yolact") -> None:
        if model_type == "yolact":
            self.lfList = ["Pa vô lê", "Đèn gầm", "Cụm đèn trước", "Đèn cửa", "Cụm gương chiếu hậu",
                            "Kính chết góc cửa", "Cụm đèn hậu", "Đèn lùi", "Kính hông", "Hông vè sau xe",
                            "Trụ kính sau", "Trụ kính trước", "Đèn xi nhan ba đờ sốc", "Đèn phản quang", "Vè trước xe",
                            "Viền nóc xe"]

            self.checklist = copy.deepcopy(self.lfList)
            self.checklist.extend(["Kính cánh cửa", "Cánh cửa"])

            self.backPartList = ["Đèn phản quang", "Trụ kính sau", "Cốp sau", "Kính chắn gió sau", "Cụm đèn hậu"]
            self.frontPartList = ["Trụ kính trước", "Vè trước xe", "Nẹp ca lăng",
                            "Kính chắn gió trước", "Ca pô trước", "Ba đờ sốc trước",
                            "Cụm đèn trước", "Ca lăng"]
        else:
            self.lfList = ['Viền nóc (mui)', 'Pa vô lê', "Đèn phản quang ba đờ sốc sau", "Đèn gầm", "Ốp đèn gầm", "Cụm đèn trước",
                           "Mặt gương (kính) chiếu hậu", "Vỏ gương (kính) chiếu hậu", "Chân gương (kính) chiếu hậu", "Đèn xi nhan trên gương chiếu hậu",
                           "Trụ kính trước", "Tai (vè trước) xe", "Ốp Tai (vè trước) xe", "Đèn xi nhan ba đ sốc", "Ốp đèn xi nhan ba đ sốc", "Đèn hậu",
                           "Kính hông (xe 7 chỗ)", "Hông (vè sau) xe", "Trụ kính sau", "Đèn hậu trên cốp sau", "Bậc cánh cửa"]

            self.FBLRAll = []

            self.FBLRList = ["Kính cánh cửa", "Cánh cửa", "Kính chết góc cửa", "Tay mở cửa", "Trụ kính cánh cửa", "La giăng (Mâm xe)", "Lốp (vỏ) xe"]
            self.FBLRAll.extend(self.FBLRList)

            self.lrFrontList = ["Đèn gầm", "Ốp đèn gầm", "Cụm đèn trước", "Mặt gương (kính) chiếu hậu", "Vỏ gương (kính) chiếu hậu",
                                "Chân gương (kính) chiếu hậu", "Đèn xi nhan trên gương chiếu hậu", "Trụ kính trước", "Tai (vè trước) xe",
                                "Ốp Tai (vè trước) xe", "Đèn xi nhan ba đ sốc", "Ốp đèn xi nhan ba đ sốc"]
            self.FBLRAll.extend(self.lrFrontList)

            self.lrBackList = ["Đèn phản quang ba đờ sốc sau", "Đèn hậu", "Hông (vè sau) xe",
                                "Trụ kính sau", "Đèn hậu trên cốp sau", "Ốp hông (vè sau) xe"]
            self.FBLRAll.extend(self.lrBackList)

            self.FBList = ["Viền nóc (mui)", "Bậc cánh cửa", "Kính hông (xe 7 chỗ)", "Pa vô lê"]
            
            self.topList = ["Nóc xe "]
            
            self.backPartList = ["Cốp sau / Cửa hậu", "Ba đờ sốc sau", "Ốp ba đờ sốc sau", 
                                 "Kính chắn gió sau", "Nẹp cốp sau"]
            
            self.frontPartList = ['Kính chắn gió trước', 'Ca pô trước', 'Nẹp Capo', 'Ca lăng', 'Lưới ca lăng',
                                  'Ba đờ sốc trước', 'Lưới ba đờ sốc', 'Ốp ba đờ sốc trước', "Nẹp ca pô trước"]
            
            self.checklist = copy.deepcopy(self.lfList)
            self.checklist.extend(self.FBLRList)
    
    
    def get_location_singal_part(self, class_name, defaul_location: str):
        if class_name in self.backPartList:
            return 'Sau'

        elif class_name in self.frontPartList:
            return 'Trước'

        elif class_name == "Nóc xe ":
            return 'Trên'
        
        elif class_name == "Lô gô":
            if "Sau" in defaul_location:
                return 'Sau'
            else:
                return 'Trước'
        
        elif class_name in self.FBList:
            if "Trái" in defaul_location:
                return 'Trái'
            else:
                return 'Phải'
        else:
            # print(f"{class_name} not in this case")
            return defaul_location
    

    def get_localtion_singal_special_part(self, segmentResult: dict, defaul_location: str, imageWeight):
        if segmentResult['class'] in self.lrFrontList:
            if 'Trái' in defaul_location or 'Phải' in defaul_location:
                segmentResult['location'] = f"{defaul_location.split(' ')[0]} - Trước"
            else:
                central_point = segmentResult['box_abs'][0] + segmentResult['box_abs'][1]
                if central_point >= imageWeight:
                    segmentResult['location'] = "Trái - Trước"
                else:
                    segmentResult['location'] = "Phải - Trước"
        elif segmentResult['class'] in self.lrBackList:
            if 'Trái' in defaul_location or 'Phải' in defaul_location:
                segmentResult['location'] = f"{defaul_location.split(' ')[0]} - Sau"
            else:
                central_point = segmentResult['box_abs'][0] + segmentResult['box_abs'][1]
                if central_point >= imageWeight:
                    segmentResult['location'] = "Phải - Sau"
                else:
                    segmentResult['location'] = "Trái - Sau"
        else:
            segmentResult['location'] = defaul_location
    
    def get_localtion_double_special_part(self, segmentResult_1: dict, segmentResult_2: dict, class_name: str, defaul_location: str):
        central_point_1 = segmentResult_1['box_abs'][0] + segmentResult_1['box_abs'][1]
        central_point_2 = segmentResult_2['box_abs'][0] + segmentResult_2['box_abs'][1]

        if class_name in self.lrFrontList:
            if central_point_1 > central_point_2:
                segmentResult_1['location'] = "Trái - Trước"
                segmentResult_2['location'] = "Phải - Trước"
            else:
                segmentResult_2['location'] = "Trái - Trước"
                segmentResult_1['location'] = "Phải - Trước"

        elif class_name in self.lrBackList:
            if central_point_2 > central_point_1:
                segmentResult_1['location'] = "Trái - Sau"
                segmentResult_2['location'] = "Phải - Sau"
            else:
                segmentResult_2['location'] = "Trái - Sau"
                segmentResult_1['location'] = "Phải - Sau"
        
        elif class_name in self.FBLRAll:
            if "Trái" in defaul_location:
                if central_point_1 < central_point_2:
                    segmentResult_1['location'] = "Trái - Trước"
                    segmentResult_2['location'] = "Trái - Sau"
                else:
                    segmentResult_2['location'] = "Trái - Trước"
                    segmentResult_1['location'] = "Trái - Sau"
            else:
                if central_point_2 < central_point_1:
                    segmentResult_1['location'] = "Phải - Trước"
                    segmentResult_2['location'] = "Phải - Sau"
                else:
                    segmentResult_2['location'] = "Phải - Trước"
                    segmentResult_1['location'] = "Phải - Sau"
        else:
            segmentResult_1['location'] = defaul_location
            segmentResult_2['location'] = defaul_location


    def get_max_min_point(self, segmentResults: list):
        backXPoint = {
            'max': 0,
            'min': 999999999
        }
        frontXPoint = {
            'max': 0,
            'min': 999999999
        }
        for segmentResultIdx in segmentResults:
            xPoint = segmentResultIdx['box_abs'][2] - segmentResultIdx['box_abs'][0]
            if segmentResultIdx['class'] in self.backPartList:
                backXPoint["max"] = xPoint if xPoint >= backXPoint["max"] else backXPoint["max"]
                backXPoint["min"] = xPoint if xPoint <= backXPoint["min"] else backXPoint["min"]
            
            if segmentResultIdx['class'] in self.frontPartList:
                frontXPoint["max"] = xPoint if xPoint >= frontXPoint["max"] else frontXPoint["max"]
                frontXPoint["min"] = xPoint if xPoint <= frontXPoint["min"] else frontXPoint["min"]

        return frontXPoint, backXPoint


    def get_locate_2_same_parts(self, segmentResults: list, partName: str, imageWeight):
        if partName not in self.checklist:
            return segmentResults

        frontXPoint, backXPoint = self.get_max_min_point(segmentResults)
        totalBackPart = 0
        totalFrontPart = 0
        checkPart = {}
        checkXPoint = {}
        for idx in range(len(segmentResults)):
            segmentResultIdx = segmentResults[idx]
            if segmentResultIdx['class'] == partName:
                checkPart[partName + f"_{idx}"] = segmentResultIdx
                checkXPoint[partName + f"_{idx}"] = segmentResultIdx['box_abs'][2] - segmentResultIdx['box_abs'][0]
        
            if segmentResultIdx['class'] in self.backPartList:
                totalBackPart = totalBackPart + 1
            
            if segmentResultIdx['class'] in self.frontPartList:
                totalFrontPart = totalFrontPart + 1

        firstPart = list(checkPart.keys())[0]
        firstIdx = int(firstPart.split("_")[-1])
        secondPart = list(checkPart.keys())[1]
        secondIdx = int(secondPart.split("_")[-1])
        if totalBackPart > totalFrontPart:
            if checkXPoint[firstPart] > checkXPoint[secondPart]:
                if partName in self.lfList:
                    checkPart[firstPart]['location'] = "trái"
                    checkPart[secondPart]['location'] = "phải"
                else:
                    if frontXPoint['max'] < backXPoint['max']:
                        checkPart[firstPart]['location'] = "sau phải"
                        checkPart[secondPart]['location'] = "trước phải"
                    else:
                        checkPart[firstPart]['location'] = "sau trái"
                        checkPart[secondPart]['location'] = "trước trái"
            
            else:
                if partName in self.lfList:
                    checkPart[firstPart]['location'] = "phải"
                    checkPart[secondPart]['location'] = "trái"
                else:
                    if frontXPoint['max'] < backXPoint['max']:
                        checkPart[firstPart]['location'] = "trước phải"
                        checkPart[secondPart]['location'] = "sau phải"
                    else:
                        checkPart[firstPart]['location'] = "trước trái"
                        checkPart[secondPart]['location'] = "sau trái"
        
        else:
            if checkXPoint[firstPart] > checkXPoint[secondPart]:
                if partName in self.lfList:
                    checkPart[firstPart]['location'] = "phải"
                    checkPart[secondPart]['location'] = "trái"
                else:
                    if backXPoint['max'] < frontXPoint['max']:
                        checkPart[firstPart]['location'] = "trước trái"
                        checkPart[secondPart]['location'] = "sau trái"
                    else:
                        checkPart[firstPart]['location'] = "trước phải"
                        checkPart[secondPart]['location'] = "sau phải"
            
            else:
                if partName in self.lfList:
                    checkPart[firstPart]['location'] = "trái"
                    checkPart[secondPart]['location'] = "phải"
                else:
                    if backXPoint['max'] < frontXPoint['max']:
                        checkPart[firstPart]['location'] = "sau trái"
                        checkPart[secondPart]['location'] = "trước trái"
                    else:
                        checkPart[firstPart]['location'] = "sau phải"
                        checkPart[secondPart]['location'] = "trước phải"
        
        segmentResults[firstIdx] = checkPart[firstPart]
        segmentResults[secondIdx] = checkPart[secondPart]
        return segmentResults
    

    def get_locate_only_1_part(self, segmentResults: list, partName: str, imageWeight):
        if partName not in self.checklist:
            return segmentResults

        frontXPoint, backXPoint = self.get_max_min_point(segmentResults)
        totalBackPart = 0
        totalFrontPart = 0
        checkPart = {}
        for idx in range(len(segmentResults)):
            segmentResultIdx = segmentResults[idx]
            if segmentResultIdx['class'] == partName:
                checkPart = segmentResultIdx
                checkIdx = idx

            if segmentResultIdx['class'] in self.backPartList:
                totalBackPart = totalBackPart + 1
            
            if segmentResultIdx['class'] in self.frontPartList:
                totalFrontPart = totalFrontPart + 1

        if totalBackPart > totalFrontPart:
            if frontXPoint['max'] < backXPoint['max']:
                if partName in self.lfList:
                    checkPart['location'] = "phải"
                else:
                    checkPart['location'] = "sau phải"
            else:
                if partName in self.lfList:
                    checkPart['location'] = "trái"
                else:
                    checkPart['location'] = "sau trái"
        else:
            if backXPoint['max'] < frontXPoint['max']:
                if partName in self.lfList:
                    checkPart['location'] = "trái"
                else:
                    checkPart['location'] = "trước trái"
            else:
                if partName in self.lfList:
                    checkPart['location'] = "phải"
                else:
                    checkPart['location'] = "trước phải"

        segmentResults[checkIdx] = checkPart
        return segmentResults


class postProcessMask:
    def __init__(self) -> None:
        self.splitParts = {
            "Ca lăng": ["Lô gô", "Lưới ca lăng"],
            "Lốp (vỏ) xe": ["La giăng (Mâm xe)"],
            "Cốp sau": ["Lô gô", "Nẹp cốp sau"],
            "Cánh cửa": ["Tay mở cửa"],
            "Ốp đèn xi nhan ba đ sốc": ["Đèn xi nhan ba đ sốc"],
            "Ốp đèn gầm": ["Đèn gầm"],
            "Trụ kính cánh cửa": ["Kính cánh cửa", "Kính chết góc cửa"],
            "Ba đờ sốc trước" : ["Đèn xi nhan ba đ sốc", "Ốp đèn xi nhan ba đ sốc", "Đèn gầm", "Ốp đèn gầm", 
                                 'Ca lăng', "Lưới ca lăng", "Lô gô"],
            "Ốp ba đờ sốc trước": ["Đèn xi nhan ba đ sốc", "Ốp đèn xi nhan ba đ sốc", "Đèn gầm", "Ốp đèn gầm", "Lưới ba đờ sốc"],
            "Ba đờ sốc sau": ["Đèn xi nhan ba đ sốc", "Ốp đèn xi nhan ba đ sốc", "Đèn phản quang ba đờ sốc sau",
                              "Nẹp cốp sau"]
        }
        self.listParts = []
        for part_total_idx in self.splitParts.keys():
            self.listParts.extend(self.splitParts[part_total_idx])

    
    def getNotAndMask(self, firstMask, secondMask, imageHeight, imageWeight):
        firstMask_zero = np.zeros((imageHeight, imageWeight), dtype=np.uint8)
        secondMask_zero = np.zeros((imageHeight, imageWeight), dtype=np.uint8)

        # ADD LOGIC
        firstMask_zero[firstMask > 0] = 1
        secondMask_zero[secondMask > 0] = 1
        out = np.bitwise_and(firstMask_zero, secondMask_zero)

        return np.bitwise_xor(out, firstMask_zero)

class getUUID:
    def __init__(self) -> None:
        self.FBLRDict = {
            "Kính cánh cửa": {
                "Trước": "IVtAaMny1ZSKI4M_d0VK-",
                "Sau": "Rn5T0cpOAKmVmrk0ke4Pq"
            }, "Cánh cửa": {
                "Trước": "9fvkQjYJEw9X74P08k1Q9",
                "Sau": "magT1W8r4rb-uN1k85HNl"
            }, "Kính chết góc cửa": {
                "Trước": "3d5g2LubM3UtU5URhlJw_",
                "Sau": "kg0Dju35Oo7DYKilRospp"
            }, "Tay mở cửa": {
                "Trước": "b_6ndwLjuzsq3YaiTjwDE",
                "Sau": "iXXOGn-DAJDLTm2q5OpGl"
            }, "Trụ kính cánh cửa": {
                "Trước": "uZJ4hNocX0WNdRaPZZn_S",
                "Sau": "FnG3Pc0_xRSLFdFJgz37p"
            }, "La giăng (Mâm xe)": {
                "Trước": "j0kppf7AppGtnFgl_LDpa",
                "Sau": "T3PDheTCwZGHEOcl2Ilac"
            }, "Lốp (vỏ) xe": {
                "Trước": "STwmpFr11eoe_L-0QEyJU",
                "Sau": "EfUyyFKtdbDHluex46Kw1"
            }}

        self.lrFrontDict = {
            "Đèn gầm": {
                "Phải": "JudiY3WJixf_dp4qSXEcP",
                "Trái": "xQOvBWGTlmEJjUgm8vPlv"
            }, "Ốp đèn gầm": {
                "Phải": "sZCvmrN84WiVHX9sdWBtO",
                "Trái": "I7-NVj1BNhg8eAdgK76l2"
            }, "Cụm đèn trước": {
                "Phải": "2RtumDUXF5lKkg4eUM1Br",
                "Trái": "NBdQses_bSYc4p3K1fVHo"
            }, "Mặt gương (kính) chiếu hậu": {
                "Phải": "PPpaZrg1CGrPnWZP-SjmP",
                "Trái": "f81UDhW5-b1LLmpJF0DNi"
            }, "Vỏ gương (kính) chiếu hậu": {
                "Phải": "zCGN_XcAPn8G4Qvvsf9zk",
                "Trái": "RxUAEejyT--O0PukAwRYN"
            }, "Chân gương (kính) chiếu hậu": {
                "Phải": "II93MzrSLw5n100bFy4Nk",
                "Trái": "VHX5-2xEsT1LGowm9NXQK"
            }, "Đèn xi nhan trên gương chiếu hậu": {
                "Phải": "T7u6969HbNHqSDO_pVpYm",
                "Trái": "LYhjoE69ZwOp0VEFG-ZW2"
            }, "Trụ kính trước": {
                "Phải": "FcjB0cORNPeJGneHSPSw7",
                "Trái": "IKzjjpHQ5mmEqzRtFsNWb"
            }, "Tai (vè trước) xe": {
                "Phải": "DtrLq1janMMK8Eec6rU6z",
                "Trái": "YqXPAleCuYdX30Ms2-Ilp"
            }, "Ốp Tai (vè trước) xe": {
                "Phải": "u_col-0ZxKDejq1EzU7fN",
                "Trái": "gBf9B5b1mPwoM4QD-Ej3g"
            }, "Đèn xi nhan ba đ sốc": {
                "Phải": "Wh2Wdo-Cqr_ni8_b_6Duv",
                "Trái": "3Xl9GXY-YJ_5_71q6yFIb"
            }, "Ốp đèn xi nhan ba đ sốc": {
                "Phải": "LGAI-I7tRXVJ0olqlUXvC",
                "Trái": "GZ_HGnM-w0uyH6hTyyHOE"
            }}

        self.lrBackList = {
            "Đèn phản quang ba đờ sốc sau": {
                "Phải": "T0WYtitTphgm9NFT4tbuC", 
                "Trái": "ZCU2GjMs7ZN1PFHpHbxwi"
            }, "Đèn hậu": {
                "Phải": "DbM2qPfwhc4LlZLu_8euj", 
                "Trái": "YCVi5zbKj1sQLt1fY_5iG"
            }, "Hông (vè sau) xe": {
                "Phải": "4bUKE9lswxBUkWStNVD1b", 
                "Trái": "ft8xznDsJjwMO7WBeg-R_"
            }, "Trụ kính sau": {
                "Phải": "k62ObvWpdHHLKKD9XqFAq", 
                "Trái": "-3GGYSQjMxxntxGMbfjTl"
            }, "Đèn hậu trên cốp sau": {
                "Phải": "TmsGa0jqULIxPmVxZmFQx", 
                "Trái": "IwuvevRjsGTueOXN0XuWz"
            }, "Ốp hông (vè sau) xe": {
                "Phải": "1QgkJv7scgMUNxzjsxjy-", 
                "Trái": "yt64OkGqswvofQfu_nBEp"
            }}

        self.FBDict = {
            "Viền nóc (mui)": "9cZsnlDB-eg5WYS45BXL9", 
            "Bậc cánh cửa": "6KnkKysGbBW2SiCCj4Whg", 
            "Pa vô lê": "wqB6JKWVUQ2ng0Frmq9WP",
            "Kính hông (xe 7 chỗ)": "r6JzbF8ofkPFAMqSFh1G1"}
        
        self.topDict = {
            "Nóc xe ": "hp_gY5jf8OpmvxvnhfUV7", 
            "Lô gô": "xNBaj39VUShzX9aV4rdis"}
        
        self.backPartDict = {
            "Cốp sau / Cửa hậu": "eT__vYUYckCtNa-pwbWmf", 
            "Ba đờ sốc sau": "UKZabECsHGdSaz9qteAXV", 
            "Ốp ba đờ sốc sau": "i50xW9ywl9J4d9wj3S7gU", 
            "Kính chắn gió sau": "VsuRTjwBJLco_jhD2dEqC", 
            "Nẹp cốp sau": "PCQzJ7NLLo5IFeQrDkeYB"}
        
        self.frontPartDict = {
            'Kính chắn gió trước': 'QFRWEnf4OnVevBzq-j3Vt', 
            'Ca pô trước': 'g2ma2BlHnrAg_dnAQmUMm', 
            'Nẹp Capo': 'Lb6q81yHqvKnR_a9X_eM5', 
            'Ca lăng': 'STc5BcShAHTbIa__7M1WW', 
            'Lưới ca lăng': 'PQP5-HHvPmzHO6woBe9af',
            'Ba đờ sốc trước': 'AlzdY6KYppg0-_UWi2tKf', 
            'Lưới ba đờ sốc': 'VTK-CYfjDrEgrpukfS2kR', 
            'Ốp ba đờ sốc trước': 'bC22iFVpICTs4CdcLILtm', 
            "Nẹp ca pô trước": "HfRM5Yuy3HriODw1iCYh9"}
        
        self.damage = {
            'Móp, bẹp(thụng)': 'zmMJ5xgjmUpqmHd99UNq3',
            'Nứt, rạn':'5IfgehKG297bQPLkYoZTw',
            'Vỡ, thủng, rách':'wMxucuruHBUupNOoVy2MF',
            'Trầy, xước':'yfMzer07THdYoCI1SM2LN'}

    def get_uuid(self, class_name: str, location: str = None) -> str:
        if class_name in self.FBLRDict.keys():
            return self.FBLRDict[class_name][location.split(" ")[-1]]

        elif class_name in self.lrFrontDict.keys():
            return self.lrFrontDict[class_name][location.split(" ")[0]]
        
        elif class_name in self.lrBackList.keys():
            return self.lrBackList[class_name][location.split(" ")[0]]
        
        elif class_name in self.FBDict.keys():
            return self.FBDict[class_name]
        
        elif class_name in self.topDict.keys():
            return self.topDict[class_name]
        
        elif class_name in self.backPartDict.keys():
            return self.backPartDict[class_name]
        
        elif class_name in self.frontPartDict.keys():
            return self.frontPartDict[class_name]

        else:
            return self.damage[class_name]