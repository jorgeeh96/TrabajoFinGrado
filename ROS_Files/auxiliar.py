import numpy as np
import tensorflow as tf
import random
import os
import cv2
import math

vv = [-1.5, -0.75, 0, 0.75, 1.5]
vv2 = [-0.75, -0.35, 0, 0.35, 0.75]
vvr = ['-0.65', '-0.35', '0.0', '0.35', '0.65']

def get_vel(vel_vec):
	pm = []
	for v in vel_vec:
		pm.append(v[0][0]*vv[0]+v[0][1]*vv[1]+v[0][3]*vv[3]+v[0][4]*vv[4])

	return(pm)

def get_vel_2(v):
	pm = v[0][0]*vv2[0]+v[0][1]*vv2[1]+v[0][3]*vv2[3]+v[0][4]*vv2[4]
	return(pm)

def direction(vel):
	ang = math.atan2(vel[1], vel[0])
	return(180*ang/math.pi)

def codify(ang, conv=9):
	a = math.floor((ang + np.pi)*conv/2/np.pi)

	if a == conv:
		a = 0

	oh_ang = []
	for cont in range(conv):
		if cont == a:
			oh_ang.append(1.0)
		else:
			oh_ang.append(0.0)

	return(oh_ang)

def difference(ang1, ang2):
	ang2p = ang2 - ang1

	if ang2p > 180:
		ang2p -= 360
	elif ang2p < -180:
		ang2p += 360

	return(-ang2p)

def rotation_matrix(ang):
	mat = np.array([[math.cos(ang), math.sin(ang)], [-math.sin(ang), math.cos(ang)]])
	return(mat)

class Neural_Network():
	def __init__(self, outputs, learning_rate, batch_size, epochs, loading=False):
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.hmeps = epochs

		imagen = tf.keras.Input(shape=(100, 120, 1), name='Depth_Image')
		conv1 = tf.keras.layers.Conv2D(32, (6, 6), activation='softmax', name='conv1')(imagen)
		pool1 = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', name='pool1')(conv1)
		conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='softmax', name='conv2')(pool1)
		pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', name='pool2')(conv2)
		flat_img = tf.keras.layers.Flatten()(pool2)
		fc_img1 = tf.keras.layers.Dense(64, activation='relu')(flat_img)
		fc_img3 = tf.keras.layers.Dense(64, activation='relu')(fc_img1)

		pos_xy = tf.keras.Input(shape=(2, 1), name='pos_xy')
		flat_pos_xy = tf.keras.layers.Flatten()(pos_xy)

		pos_rel_xy = tf.keras.Input(shape=(2, 1), name='pos_xy_rel')
		flat_pos_rel = tf.keras.layers.Flatten()(pos_rel_xy)

		posz = tf.keras.Input(shape=(1, 1), name='pos_z')
		flat_posz = tf.keras.layers.Flatten()(posz)

		lonReal = tf.keras.Input(shape=(1, 1), name='ang_real')
		flat_lon_real = tf.keras.layers.Flatten()(lonReal)

		angObjReal_cod = tf.keras.Input(shape=(2, 9, 1), name='ang_Image')
		conv1_aor = tf.keras.layers.Conv2D(16, (2, 2), activation='relu')(angObjReal_cod)
		pool1_aor = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid')(conv1_aor)
		flat_ang = tf.keras.layers.Flatten()(pool1_aor)

		pos_xyz = tf.keras.layers.concatenate([flat_pos_rel, flat_posz])
		pos_xyl = tf.keras.layers.concatenate([flat_pos_rel, flat_lon_real])
		pos_imgang_l = tf.keras.layers.concatenate([flat_ang, flat_lon_real])

		fc_xyz = tf.keras.layers.Dense(32, activation='relu')(pos_xyz)
		fc2_xyz = tf.keras.layers.Dense(32, activation='relu')(fc_xyz)

		fc_xyl = tf.keras.layers.Dense(32, activation='relu')(pos_xyl)
		fc2_xyl = tf.keras.layers.Dense(32, activation='relu')(fc_xyl)

		fc_imgang_l = tf.keras.layers.Dense(32, activation='relu')(pos_imgang_l)
		fc2_imgang_l = tf.keras.layers.Dense(32, activation='relu')(fc_imgang_l)

		all1 = tf.keras.layers.concatenate([fc2_xyl, fc2_xyz])
		all2 = tf.keras.layers.concatenate([all1, fc2_imgang_l])
		all3 = tf.keras.layers.concatenate([all2, fc_img3])
		fc_all = tf.keras.layers.Dense(128, activation='relu')(all3)
		fc2_all = tf.keras.layers.Dense(128, activation='relu', name='all1')(fc_all)
		fc3_all = tf.keras.layers.Dense(128, activation='relu', name='all2')(fc2_all)
		fc4_all = tf.keras.layers.Dense(128, activation='relu', name='all3')(fc3_all)

		# Vx,y
		con_xy = tf.keras.layers.concatenate([flat_pos_rel, fc_img3])
		con2_xy = tf.keras.layers.concatenate([fc4_all, con_xy])
		
		fc_x = tf.keras.layers.Dense(64, activation='relu')(con2_xy)
		vx = tf.keras.layers.Dense(64, activation='relu')(fc_x)
		vx_out = tf.keras.layers.Dense(outputs, activation='softmax', name='x')(vx)

		fc_y = tf.keras.layers.Dense(64, activation='relu')(con2_xy)
		vy = tf.keras.layers.Dense(64, activation='relu')(fc_y)
		vy2 = tf.keras.layers.Dense(64, activation='relu')(vy)
		vy_out = tf.keras.layers.Dense(outputs, activation='softmax', name='y')(vy2)

		# Vz
		con_z = tf.keras.layers.concatenate([pos_xyz, flat_lon_real])
		con2_z = tf.keras.layers.concatenate([fc4_all, con_z])

		fc_z = tf.keras.layers.Dense(32, activation='relu')(con2_z)
		vz = tf.keras.layers.Dense(32, activation='relu')(fc_z)
		vz_out = tf.keras.layers.Dense(outputs, activation='softmax', name='z')(vz)

		# Gz
		con_gz = tf.keras.layers.concatenate([pos_xyl, flat_lon_real])
		con2_gz = tf.keras.layers.concatenate([con_gz, flat_ang])
		con3_gz = tf.keras.layers.concatenate([fc4_all, con2_gz])
		fc_gz = tf.keras.layers.Dense(32, activation='relu')(con3_gz)
		gz = tf.keras.layers.Dense(32, activation='relu')(fc_gz)
		gz_out = tf.keras.layers.Dense(outputs, activation='softmax', name='gz')(gz)

		if loading == False:
			self.model = tf.keras.Model(inputs=[imagen, pos_rel_xy, posz, lonReal, angObjReal_cod], outputs=[vx_out, vy_out, vz_out, gz_out])
			self.model.compile(optimizer='adam',
								loss=['categorical_crossentropy','categorical_crossentropy','categorical_crossentropy', 'categorical_crossentropy'],
								metrics=['accuracy'])
		else:
			self.load()

	def training(self, xtrain, ytrain):
		self.model.fit(xtrain, ytrain, batch_size=self.batch_size, epochs=self.hmeps)
		self.save()

	def evaluate(self, xtest, ytest):
		self.model.evaluate(xtest, ytest)

	def predict(self, im, y, z, ang, ang2):
		out = self.model.predict([im.reshape(1,100,120,1), y.reshape(1,2,1), z.reshape(1,1,1), ang.reshape(1,1,1), ang2.reshape(1,2,9,1)])
		return(out)

	def save(self, model_name='model.h5'):
		self.model.save(model_name)

	def load(self, model_name='model.h5'):
		self.model = tf.keras.models.load_model(model_name)
