import numpy as np
import tensorflow as tf
import random
import os
import cv2
import math
from keras.utils import plot_model

vv = [-1.5, -0.75, 0, 0.75, 1.5]
vv2 = [-0.75, -0.35, 0.0, 0.35, 0.75]
vvr = ['-0.65', '-0.35', '0.0', '0.35', '0.65']

def read_file(filename):
	f = open(filename, 'r')

	ret_vec = []
	for lines in f.readlines():
		ret_vec.append([float(l) for l in lines.split(' ')])

	f.close()

	return(ret_vec)

def get_paths(path_way):
	l = len(os.listdir(path_way))

	ret_vec = []
	for i in range(l):
		ret_vec.append(path_way+'/'+str(i+1)+'.png')

	return(ret_vec)

def get_vel(vel_vec):
	pm = []
	for v in vel_vec[:-1]:
		pm.append(v[0][0]*vv[0]+v[0][1]*vv[1]+v[0][3]*vv[3]+v[0][4]*vv[4])
	pm.append(vel_vec[-1][0][0]*vv2[0]+vel_vec[-1][0][1]*vv2[1]+vel_vec[-1][0][3]*vv2[3]+vel_vec[-1][0][4]*vv2[4])
	return(pm)

def direction(vel):
	ang = math.atan2(vel[1], vel[0])
	return(180*ang/math.pi)

def rotation_matrix(ang):
	mat = np.array([[math.cos(ang), math.sin(ang)], [-math.sin(ang), math.cos(ang)]])
	return(mat)

def create_samples(a1, x2, y2):
	a2 = math.atan2(y2, x2)
	d = difference(a1*180/np.pi, a2*180/np.pi)
	if d > 20:
		if d > 120:
			outs = [0.0, 0.0, 0.0, 0.0, 1.0]
			b = 4
		else:
			outs = [0.0, 0.0, 0.0, 1.0, 0.0]
			b=3
	elif d < -20:
		if d < -120:
			outs = [1.0, 0.0, 0.0, 0.0, 0.0]
			b=0
		else:
			outs = [0.0, 1.0, 0.0, 0.0, 0.0]
			b=1
	else:
		outs = [0.0, 0.0, 1.0, 0.0, 0.0]
		b=2
	#print(d, outs)

	return(outs, b)
		
def difference(ang1, ang2):
	ang2p = ang2 - ang1

	if ang2p > 180:
		ang2p -= 360
	elif ang2p < -180:
		ang2p += 360

	return(-ang2p)

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

def resize_matrix(mat, mfac):
	ret = np.zeros([len(mat)*mfac, len(mat[0])*mfac], dtype=float)

	for i in range(len(ret)):
		mtd = sum(mat[math.floor(i/mfac)])

		for j in range(len(ret[0])):
			ret[i][j] = mat[math.floor(i/mfac)][math.floor(j/mfac)]/mtd

	return(ret)

def unify_images(images_set, c, tipo=False):
	if tipo == True:
		caux = c
		for i in range(7):
			if caux < 0:
				image = cv2.imread(images_set[0])
				image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				img = cv2.resize(image, (60, 50))
				img = img/255.0
				ret = np.concatenate([ret, img], -1)
			else:
				image = cv2.imread(images_set[caux])
				image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				img = cv2.resize(image, (60, 50))
				img = img/255.0

				if caux == c:
					ret = img
				else:
					ret = np.concatenate([ret, img], -1)

			caux -= 1
		image = None
	else:
		image = cv2.imread(images_set[c])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(image, (120, 100))
		ret = img/255.0


	return(ret)

class Step():
	def __init__(self, pose, velocity, img, fpose):
		prel = np.array([fpose[0]-pose[0], fpose[1]-pose[1]])
		vrel = velocity[0:2]

		R = rotation_matrix(pose[-1])
		aux_pos = np.dot(R, prel)
		aux_vel = np.dot(R, vrel)

		self.posgir = [aux_pos[0], aux_pos[1]]

		self.lon = math.sqrt(prel[0]**2 + prel[1]**2)

		uav_yaw = pose[3]

		self.pos = [prel[0], prel[1], fpose[2]-pose[2], uav_yaw]

		self.velocity = [aux_vel[0], aux_vel[1], velocity[2], velocity[3]]
		self.img = img

		self.vx, self.nx = self.one_hot_vel(self.velocity[0], 0.05, 0.75)
		self.vy, self.ny = self.one_hot_vel(self.velocity[1], 0.05, 0.75)
		self.vz, self.nz = self.one_hot_vel(self.velocity[2], 0.1, 0.5)

		self.ngz, self.mrd = self.one_hot_vel(self.velocity[3], 0.05, 0.75)
		#self.ngz, self.mrd = create_samples(self.pos[-1], self.pos[0], self.pos[1])

		self.aux = [self.nx, self.ny, self.nz, self.mrd]

	def one_hot_vel(self, val, lim=0.2, lim2=0.35):
		p = -1
		if lim2 > val > lim:
			p = 1
			ret = [0.0, 1.0, 0.0, 0.0, 0.0]
		elif -lim2 < val < -lim:
			p = 3
			ret = [0.0, 0.0, 0.0, 1.0, 0.0]
		elif val > lim2:
			p = 0
			ret = [1.0, 0.0, 0.0, 0.0, 0.0]
		elif val < -lim2:
			p = 4
			ret = [0.0, 0.0, 0.0, 0.0, 1.0]
		else:
			p = 2
			ret = [0.0, 0.0, 1.0, 0.0, 0.0]
		return(ret, p)

	def one_hot_vel_3(self, val, lim=0.2, lim2=0.35):
		p = -1
		if val > lim:
			p = 1
			ret = [1.0, 0.0, 0.0]
		elif val < -lim:
			p = 3
			ret = [0.0, 0.0, 1.0]
		else:
			p = 2
			ret = [0.0, 1.0, 0.0]
			
		return(ret, p)

class Dataset():
	def __init__(self):
		self.posx = []
		self.posy = []
		self.posz = []
		self.vel = []
		self.imgs = []
		self.oh_vel = [[],[],[],[]]
		self.dir = []
		self.ang = []
		self.double = []
		self.aux = []

	def load_data(self, data):
		prob = [[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0]]
		for obj in data:

			self.posx.append(obj.pos[0:2])			# Posicion xy
			self.posy.append(obj.posgir[0:2])		# Posicion x'y'
			self.posz.append(obj.pos[2])			# Posicion z
			self.vel.append(obj.velocity)
			self.imgs.append(obj.img)				# Imagenes
			self.ang.append(obj.lon)				# Longitud
			phi = math.atan2(obj.pos[1], obj.pos[0])
			self.double.append([codify(0), codify(difference(phi*180/np.pi, obj.pos[-1]*180/np.pi)*np.pi/180)])#obj.pos[1]
			
			self.oh_vel[0].append(obj.vx)
			self.oh_vel[1].append(obj.vy)
			self.oh_vel[2].append(obj.vz)
			self.oh_vel[3].append(obj.ngz)

			self.dir.append(direction([obj.velocity[0], obj.velocity[1]]))

			self.aux.append(obj.aux)
			prob[0][obj.aux[0]] += 1
			prob[1][obj.aux[1]] += 1
			prob[2][obj.aux[2]] += 1
			prob[3][obj.aux[3]] += 1

		# Numero de muestras de cada tipo
		print('Muestras de cada tipo:')
		print(prob)
		ret_img = np.reshape(self.imgs, [len(self.imgs),  100, 120, 1])
		ret_pos_x = np.reshape(self.posx, [len(self.posx), 2, 1])
		ret_pos_y = np.reshape(self.posy, [len(self.posy), 2, 1])
		ret_pos_z = np.reshape(self.posz, [len(self.posz), 1, 1])
		ret_ang = np.reshape(self.ang, [len(self.ang), 1, 1])
		ret_ang2 = np.reshape(self.double, [len(self.double), 2, 9, 1])

		
		return(ret_img, ret_pos_x, ret_pos_y, ret_pos_z, ret_ang, ret_ang2)

class Neural_Network():
	def __init__(self, outputs, learning_rate, batch_size, epochs, loading=False):
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.hmeps = epochs

		imagen = tf.keras.Input(shape=(100, 120, 1), name='Depth_Image')
		conv1 = tf.keras.layers.Conv2D(32, (6, 6), activation='relu', name='conv1')(imagen)
		pool1 = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', name='pool1')(conv1)
		conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv2')(pool1)
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
		con_z = tf.keras.layers.concatenate([fc2_xyz, flat_lon_real])
		con2_z = tf.keras.layers.concatenate([fc4_all, con_z])

		fc_z = tf.keras.layers.Dense(32, activation='relu')(con2_z)
		vz = tf.keras.layers.Dense(32, activation='relu')(fc_z)
		vz_out = tf.keras.layers.Dense(outputs, activation='softmax', name='z')(vz)

		# Gz
		con2_gz = tf.keras.layers.concatenate([pos_xyl, flat_ang])
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
		self.model.summary()

		plot_model(self.model, to_file='mv8.png')

	def training(self, xtrain, ytrain):
		self.model.fit(xtrain, ytrain, batch_size=self.batch_size, epochs=self.hmeps)
		self.save()

	def evaluate(self, xtest, ytest):
		self.model.evaluate(xtest, ytest)

	def predict(self, im, y, z, ang, ang2):
		out = self.model.predict([im.reshape(1,100,120,1), y.reshape(1,2,1), z.reshape(1,1,1), ang.reshape(1,1,1), ang2.reshape(1,2,9,1)])
		return(out)

	def save(self, model_name='modelo_guay_2.h5'):
		self.model.save(model_name)

	def load(self, model_name='model_vel.h5'):
		self.model = tf.keras.models.load_model(model_name)
