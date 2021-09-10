from core.gface import Gface


def demoG7photo(show_steps=False):
    gf = Gface()

    G7List = []
    for i in range(0, 1000, 100):
        G7List.append(str(7000 + i))

    # 添加人脸与更新人脸共用一个接口
    # 参数一: 学生ID
    # 参数二: 单人照文件名 (带后缀)
    for ID in G7List:
        gf.UpdateFace(ID, ID + ".jpg")

    # 此处添加一个不存在的ID作为干扰项
    # 同时ID为7900的人物并未出现在图中
    G7List.append('2300')

    # 合照识别为单独接口
    # 参数一: 总ID列表 (注册课程的学生名单对应的ID列表)
    # 参数二: 合照文件名 (带后缀)
    ra, rb, rc = gf.DetectWithList(
        id_list=G7List,
        photo_file_name="photo3.jpg",
        show_steps=show_steps
    )

    # 返回值 a 为 在总ID列表中 识别到的人脸的ID列表 (在名单里 也在合照里的)
    # 返回值 b 为 在总ID列表中 没有识别到的人脸的ID列表 (在名单里 不在合照里的)
    # 返回值 c 为 在总ID列表中 但是尚未登记到GFace的人脸的ID列表 (在名单里 但是Gface没有登记过的)
    print("Found: ", ra)
    print("Not found: ", rb)
    print("No data: ", rc)


def demoUSphoto(show_steps=False):
    gf = Gface()

    USlist = [
        '7300',
        '7900',
        '8000',
        '2700'
    ]

    for ID in USlist:
        res = gf.UpdateFace(ID, ID + '.jpg')
        print(ID, res)
        # temp_img中没有ID 2700 的照片 故返回False 其他返回True

    ra, rb, rc = gf.DetectWithList(
        id_list=USlist,
        photo_file_name="photo1.jpg",
        show_steps=show_steps
    )

    print("Found: ", ra)
    print("Not found: ", rb)
    print("No data: ", rc)


def demoG20Photo(show_steps=False):
    gf = Gface()

    # 这个demo用于演示直接将照片进行全数据库匹配的功能
    ra = gf.DetectWithoutList(
        photo_file_name="photo4.jpg",
        show_steps=show_steps
    )

    print("Found: ", ra)


if __name__ == '__main__':
    # 照片需要为jpg格式
    # 照片需要先上传到 GFaceTool/temp_img/下面 运行结束之后可以清空
    # show_steps选项会显示出检测过程中的每一个具体步骤

    # demoG7photo(show_steps=True)

    demoUSphoto(show_steps=True)

    # demoG20Photo(show_steps=False)

