import numpy as np
import pandas as pd
import editdistance
# import gensim
from neo4j import GraphDatabase, basic_auth,kerberos_auth,custom_auth,TRUST_ALL_CERTIFICATES

'''
需要先构建好图谱才能进行融合操作
'''

#链接数据库
driver = GraphDatabase.driver("neo4j://localhost:7687", auth=basic_auth("neo4j","admin"), encrypted=False)
session = driver.session()
csv_data = pd.read_csv("./data/DBLP-Scholar_perfectMapping.csv")
csv_data['idDBLP'] = csv_data['idDBLP'].map(lambda x: str(x))
# #加载预训练模型
# word2vec = gensim.models.KeyedVectors.load_word2vec_format("./data/GoogleNews-vectors-negative300.bin.gz", binary=True)
#读取Neo4j的值
data_DBLP_paper = session.run("MATCH (DBLP_paper:DBLP_paper) return DBLP_paper.DBLP_title,DBLP_paper.DBLP_id,DBLP_paper.year ")
data_Scholar_paper = session.run("MATCH (Scholar_paper:Scholar_paper) return Scholar_paper.Scholar_title,Scholar_paper.Scholar_id,Scholar_paper.year,Scholar_paper.remark ")

#初始化
dlists = []
dlists_id = []
dlists_year = []
dlists_t = []
slists = []
slists_id = []
slists_year = []
slists_t = []
slists_remark = []
zero = 0

#把DBLP的作者结点导入dlist
for d in data_DBLP_paper:
    bs = str(d[0])
    bs_id = str(d[1])
    bs_year = str(d[2])
    dlists.append(bs) 
    dlists_id.append(bs_id)
    dlists_year.append(bs_year)


#把Scholar的作者结点导入slist
for d in data_Scholar_paper:
    bs = str(d[0])
    bs_id = str(d[1])
    bs_year = str(d[2])
    bs_remark = str(d[3])
    slists.append(bs)
    slists_id.append(bs_id)
    slists_year.append(bs_year)
    slists_remark.append(bs_remark)
    
#编辑距离函数
def edit_distance(word1, word2):
    # len1 = len(word1)
    # len2 = len(word2)
    # dp = np.zeros((len1 + 1, len2 + 1))
    # for i in range(len1 + 1):
    #     dp[i][0] = i
    # for j in range(len2 + 1):
    #     dp[0][j] = j
    # for i in range(1, len1 + 1):
    #     for j in range(1, len2 + 1):
    #         delta = 0 if word1[i - 1] == word2[j - 1] else 1
    #     dp[i][j] = min(dp[i-1][j-1] + delta, min(dp[i-1][j] + 1, dp[i][j-1] + 1))
    # return dp[len1][len2]
    distance = editdistance.eval(word1,word2)
    return distance

#杰卡德函数
def Jaccrad(terms_model,reference):
    grams_reference = set(reference)
    grams_model = set(terms_model)
    temp = 0
    for i in grams_reference:
        if i in grams_model:
            temp = temp + 1
    fenmu = len(grams_model) + len(grams_reference) - temp
    jaccard_coefficient = float(temp/fenmu)
    return jaccard_coefficient

#查询DBLP文章对应的作者
def DBLP_author(a):
    DBLP_paper_author = session.run("MATCH p=(m)-[r:DBLP_publish]->(DBLP_paper{DBLP_id:'%s'}) return m.name" % (a))
    for d in DBLP_paper_author:
        bs = str(d[0])
        dlists_t.append(bs)
    return dlists_t

#查询Scholar文章对应的作者
def Scholar_author(a):
    Scholar_paper_author = session.run("MATCH p=(m)-[r:Scholar_publish]->(Scholar_paper{Scholar_id:'%s'}) return m.name" % (a))
    for d in Scholar_paper_author:
        bs = str(d[0])
        slists_t.append(bs)
    return slists_t

#查询DBLP的会议信息
def DBLP_Venue(a):
    DBLP_Venue = session.run("MATCH p=(DBLP_paper{DBLP_id:'%s'})-[r:DBLP_included]->(m) return m.DBLP_V_ID" % (a))
    return DBLP_Venue

#查询Scholar的会议的信息
def Scholar_Venue(a):
    Scholar_venue = session.run("MATCH p=(Scholar_paper{DBLP2_id:'%s'})-[r:Scholar_included]->(m) return m.Scholar_V_ID" % (a))
    return Scholar_venue

#分词函数     
def spilt_sentence(sentence):
    words = sentence.spilt()
    return words

TP = 0.001
FP = 0.001
FN = 5347
step = 0
tag = True
Flag = True #查询最佳匹配是否正确标记

#把数据拿出来两两比较    
for i in range(len(dlists)):
    for j in range(len(slists)):
        a = dlists[i]
        b = slists[j]
        DBLP_id = dlists_id[i]
        Scholar_id = slists_id[j]
        DBLP_year = int(dlists_year[i])
        if slists_year[j] != 'null':
            temp = float(slists_year[j])
            Scholar_year = int(temp)
        #计算相似度
        jacd = Jaccrad(a,b)
        std = edit_distance(a,b)/max(len(a),len(b))
        edit = 1-std
        #计算平均值
        huizon = (jacd+edit)/2
        #判断
        if 0.70 < huizon < 1.1:
            #查找在匹配集中的数据
            data = csv_data.loc[csv_data['idDBLP'] == DBLP_id, 'idScholar']
             #判断标记
            tag = True
            #判断进行匹配
            if Scholar_year != 'null':
                if DBLP_year == Scholar_year:
                    tag = True
                else:
                    tag = False
            #循环查询数组判断是否正确
            if data.shape != (0,):
                for m in range(len(data)):
                    data_t = str(data.values[m])
                    if data_t == Scholar_id:
                        Flag = True
                        break
                    else:
                        Flag = False
            
                if Flag and tag:
                    TP+=1
                    #生成匹配成功的表格
                    csv_data.loc[csv_data['idDBLP'] == DBLP_id, 'Scholar_id'] = 'CORRECT!!!!!!'
                    #print("<DBLP_name,Scholar_name>-<%s,%s>-<%s,%s>|平均相似度: %f " % (dlists[i], slists[j], dlists_id[i],slists_id[j],huizon))
                    #----------------------------------融合功能------------------------------------------------
                    # #文章结点融合
                    # try:
                    #     session.run(" MATCH (a1:ACM_paper {ACM_id:'%s'}),(a2:DBLP2_paper {DBLP2_id:'%s'}) WITH head(collect([a1,a2])) as nodes CALL apoc.refactor.mergeNodes(nodes,{properties:'combine', mergeRels:true}) yield node return nodes" % (ACM_id,DBLP2_id))
                    # except:
                    # #文章对应的会议结点融合
                    #     A_venue_ID = ACM_Venue(ACM_id)
                    #     D_venue_ID = DBLP2_Venue(DBLP2_id)
                    #     session.run("MATCH (a1:ACM_venue {venue:'%s'}),(a2:DBLP2_venue {venue:'%s'}) WITH head(collect([a1,a2])) as nodes CALL apoc.refactor.mergeNodes(nodes,{properties:'combine', mergeRels:true}) yield node return nodes" % (A_venue_ID,D_venue_ID))
                    #     #查询文章对应的作者
                    #     alists_t = ACM_author(ACM_id)
                    #     dlists_t = DBLP2_author(DBLP2_id)
                    #     print(alists_t,dlists_t)
                    #     #文章对应的作者的结点融合,在文章对应的数据集中进行匹配，大于0.5的就进行融合
                    #     for n in range(len(alists_t)):
                    #         for m in range(len(dlists_t)):
                    #             aa = alists_t[n]
                    #             bb = dlists_t[m]
                    #             #计算作者结点的相似度
                    #             jacd_t = Jaccrad(aa,bb)
                    #             std_t = edit_distance(aa,bb)/max(len(aa),len(bb))
                    #             edit_t = 1-std_t
                    #             huizon_t = (jacd_t+edit_t)/2
                    #             #平均相似度大于0.69就进行融合
                    #             if huizon_t > 0.69:
                    #                 print('-------------------------即将做融合的作者结点')
                    #                 print(huizon_t)
                    #                 print(aa,bb)
                    #                 try:
                    #                     session.run("MATCH (a1:ACM_author {name:'%s'}),(a2:DBLP2_author {name:'%s'}) WITH head(collect([a1,a2])) as nodes CALL apoc.refactor.mergeNodes(nodes,{properties:'combine', mergeRels:true}) yield node return nodes " % (aa,bb))
                    #                 except:
                    #                     pass
                        # #清空list
                        # alists_t.clear()
                        # dlists_t.clear()
                    #--------------------------------融合功能-----------------------------------------                
                else:
                    FP+=1
                    print("<DBLP_name,Scholar_name>-<%s,%s>|平均相似度: %f " % (a, b, huizon))
                    print('------------------------不正确嗷！')
                    #print(ACM_id, DBLP2_id)
                    csv_data.loc[csv_data['idDBLP'] == DBLP_id, 'idScholar'] = 'ERROR!!!!'
                    csv_data.loc[csv_data['idDBLP'] == DBLP_id, 'DBLP_name'] = a
                    csv_data.loc[csv_data['idDBLP'] == DBLP_id, 'Scholar_name'] = b
            else:
                data = 'xxxx'
        #每一步的情况
        step = step + 1
        if step % 1000000 == 0:
            Perecision_t = TP/(TP+FP)
            print('------------------------------------第 %d 精确度是 %f ：' % (step,Perecision_t))

Perecision = TP/(TP+FP)
FN = FN - TP
Recall = TP/(TP+FN)
print('最终精确度：', Perecision)
print('最终召回率：', Recall)
#生成匹配错误日志
csv_data.to_csv('DBLP_Scholar_matching_excel.csv', encoding='utf-8', index=0)





