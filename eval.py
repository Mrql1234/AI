import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# 模型目录
CHECKPOINT_DIR = './runs/1589975137/checkpoints'
INCEPTION_MODEL_FILE = 'model/tensorflow_inception_graph.pb'

# inception-v3模型参数
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # inception-v3模型中代表瓶颈层结果的张量名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  # 图像输入张量对应的名称

# 测试数据
#file_path = './data/flower_photos/roses/295257304_de893fc94d.jpg'
file_path = './data/flower_photos/2.jpg'

#file_path = './data/flower_photos/tulips/146884869_b1a8fa9c4e_n.jpg'
# file_path = './data/flower_photos/roses/12240303_80d87f77a3_n.jpg'
# file_path = './data/flower_photos/dandelion/7355522_b66e5d3078_m.jpg'
# file_path = './data/flower_photos/dandelion/16159487_3a6615a565_n.jpg'
# file_path = './data/flower_photos/sunflowers/6953297_8576bf4ea3.jpg'
# file_path = './data/flower_photos/sunflowers/40410814_fba3837226_n.jpg'
# file_path = './data/flower_photos/tulips/11746367_d23a35b085_n.jpg'
y_test = []

# 读取数据
image_data = tf.gfile.GFile(file_path, 'rb').read()
#展示图片
def plot_image(image,all_predictions,all_predictions2):
    img = tf.image.decode_jpeg(image)
    img.array = img.eval()
    for i in range(len(all_predictions2)): 
        num = all_predictions2[i]
        if num == 0:
            label = "tulips" + str(all_predictions[i][num])        
        elif num == 1:
            label = "dandelion" + str(all_predictions[i][num])
        elif num == 2:
            label = "rose" + str(all_predictions[i][num])
        elif num == 3:
            label = "sunflowers" + str(all_predictions[i][num])
        else: 
            label = "daisy" + str(all_predictions[i][num])   
    plt.title(label)
    plt.imshow(img.array)
    plt.axis('off')
    plt.show()

# 评估
checkpoint_file = tf.train.latest_checkpoint(CHECKPOINT_DIR)
with tf.Graph().as_default() as graph:
    with tf.Session().as_default() as sess:
        # 读取训练好的inception-v3模型
        with tf.gfile.GFile(INCEPTION_MODEL_FILE, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # 加载inception-v3模型，并返回数据输入张量和瓶颈层输出张量
        bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
            graph_def,
            return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

        # 使用inception-v3处理图片获取特征向量
        bottleneck_values = sess.run(bottleneck_tensor,
                                     {jpeg_data_tensor: image_data})
        # 将四维数组压缩成一维数组，由于全连接层输入时有batch的维度，所以用列表作为输入
        bottleneck_values = [np.squeeze(bottleneck_values)]

        # 加载元图和变量
        saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # 通过名字从图中获取输入占位符
        input_x = graph.get_operation_by_name(
            'BottleneckInputPlaceholder').outputs[0]

        #定义keep_prob
        keep_prob = graph.get_operation_by_name('kp').outputs[0]        

        # 我们想要评估的tensors   evaluation/ArgMax得到的是最大索引值  
        predictions = graph.get_operation_by_name('final_training_ops/Softmax').outputs[0]
        predictions2 = graph.get_operation_by_name('evaluation/ArgMax').outputs[0]

        
        # 收集预测值
        all_predictions = []
        all_predictions2 = []
        all_predictions,all_predictions2 = sess.run([predictions,predictions2], {input_x: bottleneck_values,keep_prob: 1})
        
		#for op in graph.get_operations():
            #print(op.name,":",op.values())i


print(all_predictions2)
print(all_predictions)

with tf.Session() as sess:
    plot_image(image_data, all_predictions, all_predictions2)

# 如果提供了标签则打印正确率
#if y_test is not None:
    #correct_predictions = float(sum(all_predictions == y_test))
    #print('\nTotal number of test examples: {}'.format(len(y_test)))
    #print('Accuracy: {:g}'.format(correct_predictions / float(len(y_test))))
