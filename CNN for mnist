'''
from book 'Tensorflow+Keras 深度学习人工智能实践应用'
TensorFlow
MNIST数据集，卷积神经网络
'''
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def weight(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1),name='W')
def bias(shape):
    return tf.Variable(tf.constant(0.1,shape=shape),name='b')
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

with tf.name_scope('optimizer'):
    y_label=tf.placeholder("float",shape=[None,10],name="y_label")
    loss_function=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_predict,labels=y_label))
    optimizer=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_function)
    
with tf.name_scope("evaluate_model"):
    correct_prediction=tf.equal(tf.argmax(y_predict,1),tf.argmax(y_label,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
    
trainEpochs = 30
batchSize = 100
totalBatchs = int(mnist.train.num_examples/batchSize)
epoch_list=[];accuracy_list=[];loss_list=[];
from time import time
startTime=time()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(trainEpochs):
    for i in range(totalBatchs):
        batch_x, batch_y = mnist.train.next_batch(batchSize)
        sess.run(optimizer,feed_dict={x: batch_x,
                                      y_label: batch_y})
    loss,acc=sess.run([loss_function,accuracy],feed_dict={x:mnist.validation.images,
                                                        y_label:mnist.validation.labels})
    epoch_list.append(epoch);loss_list.append(loss);accuracy_list.append(acc)
    
    print("Trian Epoch:",'%02d'%(epoch+1),"Loss=","{:.9f}".format(loss),"Accuracy:",acc)

duration=time()-startTime
print("Train Finished takes:",duration)   

#the loss about epoch
%matplotlib inline
import matplotlib.pyplot as plt
fig=plt.gcf()
fig.set_size_inches(10,5)
plt.plot(epoch_list,loss_list,label='loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper left') 

#the accuracy about epoch
plt.plot(epoch_list, accuracy_list,label="accuracy" )
fig = plt.gcf()
fig.set_size_inches(10,5)
plt.ylim(0.8,1)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()

#test model in test datasets of mnist 
len(mnist.test.images)
print("Accuracy:",sess.run(accuracy,feed_dict={x:mnist.test.images,y_label:mnist.test.labels}))

prediction_result=sess.run(tf.argmax(y_predict,1),feed_dict={x:mnist.test.images[:50],y_label:mnist.test.labels})
print(prediction_result[:20])

#result show in photoes
import numpy as np
def show_images_labels_predict(images,labels,prediction_result):
    fig=plt.gcf()
    fig.set_size_inches(8,10)
    for i in range(0,20):
        ax=plt.subplot(5,5,1+i)
        ax.imshow(np.reshape(images[i],(28,28)),cmap='binary')
        ax.set_title("label=" +str(np.argmax(labels[i]))+
                     ",predict="+str(prediction_result[i])
                     ,fontsize=9)
    plt.show()   
show_images_labels_predict(mnist.test.images,mnist.test.labels,prediction_result)

#save the model
saver=tf.train.Saver()
save_path=saver.save(sess,"saveModel/CNN_model1")
print("Model saved in file: %s" % save_path)

#export to tensorboard
merged=tf.summary.merge_all()
train_writer=tf.summary.FileWriter('log/CNN',sess.graph)
