#!/usr/bin/env python
import rospy
import curses
import time
import math
import collections
from curses import textpad
from uav_abstraction_layer.srv import TakeOff, Land
from geometry_msgs.msg import TwistStamped, PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import pygame
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from auxiliar import *
import sys

class Boton():
	def __init__(self, initial_value=False):
		self.act = initial_value
		self.block = False

	def interaction(self, a):
		if a == True and not self.block:
			if self.act == False:
				self.act = True
			else:
				self.act = False

			self.block = True
		elif a == False and self.block == True:
			self.block = False

class Movimiento():
	def __init__(self, vmin, vmax, val, drate):
		self.min = vmin
		self.max = vmax
		self.val = val
		self.drate = drate

		self.act = False

	def set_val(self, val):
		if val > self.max:
			self.val = self.max
		elif val < self.min:
			self.val = self.min
		else:
			self.val = val

	def step(self, a, d):
		if self.act == True:
			if d == True:
				self.act = False
		else:
			if a == True:
				self.act = True

class AxisMov():
	def __init__(self, rate, smooth, name):
		self.mov_up = Movimiento(0, 1.5, 0.0, 0.2)
		self.mov_down = Movimiento(0, 1.5, 0.0, 0.2)
		self.rate = rate
		self.smooth = smooth
		self.value = 0.0

	def step(self, au, ad, du, dd):
		self.mov_up.step(au, du)
		self.mov_down.step(ad, dd)

		if self.mov_up.act == True and self.mov_down.act == True:
			pass

		elif self.mov_up.act == True and self.mov_down.act == False:
			if self.value > 0.0:
				self.mov_up.set_val(self.mov_up.val + self.rate)
				self.mov_down.set_val(0.0)
				self.value = self.mov_up.val
			elif self.value < 0.0:
				self.mov_up.set_val(0.0)
				if self.smooth == True:
					self.mov_down.set_val(self.mov_down.val - self.rate*2)
				else:
					self.mov_down.set_val(0.0)
				self.value = -self.mov_down.val
			elif self.value == 0.0:
				self.mov_up.set_val(self.rate)
				self.value = self.mov_up.val

		elif self.mov_up.act == False and self.mov_down.act == True:
			if self.value > 0.0:
				if self.smooth == True:
					self.mov_up.set_val(self.mov_up.val - self.rate*2)
				else:
					self.mov_up.set_val(0.0)
				self.mov_down.set_val(0.0)
				self.value = self.mov_up.val
			elif self.value < 0.0:
				self.mov_up.set_val(0.0)
				self.mov_down.set_val(self.mov_down.val + self.rate)
				self.value = -self.mov_down.val
			elif self.value == 0.0:
				self.mov_up.set_val(self.rate)
				self.value = -self.mov_up.val

		elif self.mov_up.act == False and self.mov_down.act == False:
			if self.value > 0.0:
				if self.smooth == True:
					self.mov_up.set_val(self.mov_up.val - self.rate*2)
				else:
					self.mov_up.set_val(0.0)
				self.mov_down.set_val(0.0)
				self.value = self.mov_up.val
			else:
				self.mov_up.set_val(0.0)
				if self.smooth == True:
					self.mov_down.set_val(self.mov_down.val - self.rate*2)
				else:
					self.mov_down.set_val(0.0)
				self.value = -self.mov_down.val

		return(self.value)

class CentroMando():
	def __init__(self, width=200, height=350, name='Operador'):
		self.width = width
		self.height = height
		self.finished = False

		self.colors = {'black': (0, 0, 0),
						'white': (255, 255, 255),
						'red': (255, 0, 0),
						'green': (0, 255, 0),
						'blue': (0, 0, 255)}

		self.initialize_window(name)

		self.dof = 4

		self.movs = []
		self.movs.append(AxisMov(0.1, False, 'glgr'))
		self.movs.append(AxisMov(0.1, True, 'updown'))
		self.movs.append(AxisMov(0.1, True, 'fwbc'))
		self.movs.append(AxisMov(0.1, True, 'leri'))

		self.matrix = np.zeros((21, 21), dtype=int)
		self.pmatrix = [10, 10, 0]
		self.reset_matrix(self.pmatrix)

		self.taking_off = False
		self.artificial = Boton(False)

	def reset_matrix(self, p):
		for i in range(21):
			self.matrix[0][i] = 1
			self.matrix[20][i] = 1
			self.matrix[i][0] = 1
			self.matrix[i][20] = 1

		self.matrix[p[0]][p[1]] = 1

	def initialize_window(self, name):
		pygame.init()
		self.gameDisplay = pygame.display.set_mode((self.width, self.height))
		pygame.display.set_caption(name)

		self.clock = pygame.time.Clock()

	def events(self):
		kdown = [False, False, False, False, False, False, False, False]
		kup = [False, False, False, False, False, False, False, False]
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self.finished = True

			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_a:
					kdown[0] = True
				elif event.key == pygame.K_d:
					kdown[1] = True
				if event.key == pygame.K_w:
					kdown[2] = True
				elif event.key == pygame.K_s:
					kdown[3] = True
				if event.key == pygame.K_UP:
					kdown[4] = True
				elif event.key == pygame.K_DOWN:
					kdown[5] = True
				if event.key == pygame.K_RIGHT:
					kdown[7] = True
				elif event.key == pygame.K_LEFT:
					kdown[6] = True

			if event.type == pygame.KEYUP:
				if event.key == pygame.K_a:
					kup[0] = True
				elif event.key == pygame.K_d:
					kup[1] = True
				if event.key == pygame.K_w:
					kup[2] = True
				elif event.key == pygame.K_s:
					kup[3] = True
				if event.key == pygame.K_UP:
					kup[4] = True
				elif event.key == pygame.K_DOWN:
					kup[5] = True
				if event.key == pygame.K_RIGHT:
					kup[7] = True
				elif event.key == pygame.K_LEFT:
					kup[6] = True

		return(kdown, kup)
		
	def step(self, angs, m):
		self.gameDisplay.fill(self.colors['white'])

		art = False

		kd, ku = self.events()
		if m == None:
			m = [0.0, 0.0, 0.0, 0.0]
			m[0] = self.movs[0].step(kd[0], kd[1], ku[0], ku[1])
			m[1] = self.movs[1].step(kd[2], kd[3], ku[2], ku[3])
			m[2] = self.movs[2].step(kd[4], kd[5], ku[4], ku[5])
			m[3] = self.movs[3].step(kd[6], kd[7], ku[6], ku[7])
		
		pygame.draw.rect(self.gameDisplay, self.colors['black'], [20, 50, 20, -int(m[2]*20)])
		pygame.draw.rect(self.gameDisplay, self.colors['black'], [60, 50, 20, int(m[3]*20)])
		pygame.draw.rect(self.gameDisplay, self.colors['black'], [100, 50, 20, -int(m[1]*20)])
		pygame.draw.rect(self.gameDisplay, self.colors['black'], [140, 50, 20, int(m[0]*20)])

		for i, val in enumerate(angs):
			for j, v in enumerate(val):
				if int(v) == 0:
					pygame.draw.rect(self.gameDisplay, self.colors['black'], [10+20*j, 100+20*i, 20, 20])
				else:
					pygame.draw.rect(self.gameDisplay, self.colors['green'], [10+20*j, 100+20*i, 20, 20])

		for i, val in enumerate(self.matrix):
			for j, v in enumerate(val):
				if v == 1:
					pygame.draw.rect(self.gameDisplay, self.colors['black'], [5+i*8, 150+j*8, 8, 8])

		if self.taking_off == True:
			self.taking_off = None

		if self.taking_off == False:
			pygame.draw.rect(self.gameDisplay, self.colors['red'], [180, 150, 15, 15])
		else:
			pygame.draw.rect(self.gameDisplay, self.colors['green'], [180, 150, 15, 15])

		if self.artificial.act == False:
			pygame.draw.rect(self.gameDisplay, self.colors['red'], [180, 175, 15, 15])
		else:
			pygame.draw.rect(self.gameDisplay, self.colors['green'], [180, 175, 15, 15])

		if pygame.mouse.get_pressed()[0]:
			mouse = pygame.mouse.get_pos()
			if 5 < mouse[0] < 5+8*21 and 150 < mouse[1] < 150+8*21:
				self.pmatrix[0] = math.floor((mouse[0] - 5)/8)
				self.pmatrix[1] = math.floor((mouse[1] - 150)/8)
				self.pmatrix[2] = 2

			if 180 < mouse[0] < 195 and 150 < mouse[1] < 165 and self.taking_off == False:
				self.taking_off = True

			if 180 < mouse[0] < 195 and 175 < mouse[1] < 190:
				art = True

		self.artificial.interaction(art)

		pygame.draw.rect(self.gameDisplay, self.colors['red'], [5+self.pmatrix[0]*8, 150+self.pmatrix[1]*8, 8, 8])

			
		ret = [self.pmatrix[0]-10, -(self.pmatrix[1]-10), self.pmatrix[2]]

		pygame.display.update()
		self.clock.tick(30)

		if self.finished == True:
			pass

		return(m, self.taking_off, self.artificial.act, ret)

class MandoVel():
	def __init__(self):
		self.uav_pose = PoseStamped()
		self.bridge = CvBridge()
		self.uav_yaw = 0.0
		self.pose_sub = rospy.Subscriber('ual/pose', PoseStamped, self.pose_callback)
		self.depth_pub = rospy.Subscriber('mbzirc_1/camera_0/depth/image_raw', Image, self.depth_callback)
		self.velocity_pub = rospy.Publisher('ual/set_velocity', TwistStamped, queue_size=1)
		self.cont = 0
		self.depth = []
		take_off_url = 'ual/take_off'
		self.take_off = rospy.ServiceProxy(take_off_url, TakeOff)
		self.to_ai = False
		self.to_cmd = False

		self.un = []

	def pose_callback(self, data):
		self.uav_pose = data

		self.aux_yaw = 2.0 * math.atan2(data.pose.orientation.z, data.pose.orientation.w)
		if np.pi < self.aux_yaw <= 2*np.pi:
			self.uav_yaw = self.aux_yaw - 2*np.pi
		elif 0 <= self.aux_yaw <= np.pi:
			self.uav_yaw = self.aux_yaw
		elif -np.pi > self.aux_yaw >= -2*np.pi:
			self.uav_yaw = self.aux_yaw + 2*np.pi
		elif 0 >= self.aux_yaw >= -np.pi:
			self.uav_yaw = self.aux_yaw

	def depth_callback(self, data):
		self.depth = self.bridge.imgmsg_to_cv2(data, "passthrough")
		self.deep, self.deep_save = self.converter(self.depth, self.to_ai)

	def converter(self, img, res=True):

		minval = 1000
		maxval = 0

		imagen = np.zeros((len(img), len(img[0])), dtype=float)
		img_aux = np.copy(img)

		aux = np.isnan(img)
		img_aux[aux] = 1000
		minval = img_aux.min()
		img_aux[aux] = -100
		maxval = img_aux.max()

		imagen = (img-minval)*255/(maxval-minval)
		aux = np.isnan(imagen)
		imagen[aux] = 255

		img_mov = np.copy(imagen)
		if res == True:
			img_mov = cv2.resize(img_mov, (120, 100))
			img_mov = img_mov/255.0

		return(img_mov, imagen)

	def save_moment(self):
		self.cont += 1
		cv2.imwrite('./depth_images/'+str(self.cont)+'.png', self.deep_save)
		f = open('velocity.out', 'a')
		f.write('{:.3f} {:.3f} {:.3f} {:.3f}\n'.format(self.vel[0], self.vel[1], self.vel[2], self.vel[3]))
		f.close()

		f = open('pose.out', 'a')
		f.write('{:.3f} {:.3f} {:.3f} {:.3f}\n'.format(self.uav_pose.pose.position.x, self.uav_pose.pose.position.y, self.uav_pose.pose.position.z, self.uav_yaw))
		f.close()

	def unify_images(self):
		if self.un == []:
			self.un = ret = self.deep
			for i in range(6):
				self.un = np.concatenate([self.un, self.deep], -1)

		else:
			self.un = np.concatenate([self.deep, self.un[:, :-60],], -1)



	def execute(self, final_point):
		rate = rospy.Rate(10)
		cm = CentroMando()
		fpoint = final_point
		ang_img = [codify(0), codify(self.uav_yaw)]

		flag_cmd = False
		flag_ai = False
		z_takeoff = 1.5

		if len(sys.argv) > 1:
			args = sys.argv[1:]
			finished = False
			kont = 0
			posaux = [0.0, 0.0, 0]
			
			while not finished:

				if args[kont] == 'x':
					flag_ai = True
					posaux[0] = float(args[kont+1])
					kont += 2
					if kont >= len(args):
						finished = True
				elif args[kont] == 'y':
					flag_ai = True
					posaux[1] = float(args[kont+1])
					kont += 2
					if kont >= len(args):
						finished = True
				elif args[kont] == 'z':
					flag_ai = True
					posaux[2] = float(args[kont+1])
					kont += 2
					if kont >= len(args):
						finished = True
				elif args[kont] == 'takeoff':
					self.to_cmd = True
					flag_cmd = True
					kont += 1
					cm.taking_off = True
					if kont >= len(args):
						finished = True
				elif args[kont] == 'h':
					z_takeoff = float(args[kont+1])
					kont += 2
					if kont >= len(args):
						finished = True
				else:
					finished = True

			if flag_ai == True:
				flag_cmd = True
				self.to_cmd = True
				self.to_ai = True

			fpoint = posaux

		contador = 0

		while (not cm.finished) and (contador < 450):
			# Conador para terminar en un determinado momento
			#contador += 1
			rate.sleep()


			if self.to_cmd == False:
				m, self.to_cmd, self.to_ai, fpoint = cm.step(ang_img, None)

			if self.to_cmd == True:
				self.take_off(z_takeoff, True)
				if self.to_ai == True:
					vels = Neural_Network(5, 0.001, 32, 5, True)
				self.to_cmd = None

			if self.to_ai == True and self.to_cmd == None:

				posex = fpoint[0]-self.uav_pose.pose.position.x
				posey = fpoint[1]-self.uav_pose.pose.position.y
				aux_pos = np.array([posex, posey])
				R = rotation_matrix(self.uav_yaw)
				pos = np.dot(R, aux_pos)

				posez = fpoint[2]-self.uav_pose.pose.position.z
				ang_env = math.atan2(posey, posex)
				lon_env = math.sqrt(posex**2+posey**2)
				ang_img = [codify(0), codify(difference(ang_env*180/np.pi, self.uav_yaw*180/np.pi)*np.pi/180.0)]
				
				f = open('ref.out', 'a')
				f.write('{:.3f} {:.3f} {:.3f}\n'.format(fpoint[0], fpoint[1], fpoint[2]))
				f.close()

				m = vels.predict(np.array(self.deep), pos, np.array(posez), np.array(lon_env), np.array(ang_img))
				m2 = get_vel(m[:-1])
				g2 = get_vel_2(m[-1])
			
				vel_cmd = TwistStamped()
				vel_cmd.header.stamp = rospy.Time.now()
				vel_cmd.header.frame_id = 'map'
				self.vel = [0, 0, 0, 0]
				self.vel[0] = m2[0]*math.cos(self.uav_yaw) - m2[1]*math.sin(self.uav_yaw)
				self.vel[1] = m2[0]*math.sin(self.uav_yaw) + m2[1]*math.cos(self.uav_yaw)

				self.vel[2] = m2[2]
				self.vel[3] = g2
				_, self.to_cmd, self.to_ai, fpoint = cm.step(ang_img, [g2, m2[2], m2[0], m2[1]])
				if flag_cmd == True:
					self.to_cmd = None
				if flag_ai == True:
					self.to_ai = True
					fpoint = posaux
				vel_cmd.twist.linear.x = -self.vel[0]
				vel_cmd.twist.linear.y = -self.vel[1]
				vel_cmd.twist.linear.z = -self.vel[2]
				vel_cmd.twist.angular.z = -self.vel[3]
				self.velocity_pub.publish(vel_cmd)
				self.save_moment()

			elif self.to_cmd == None and self.to_ai == False:
				ang_img = [codify(0), codify(self.uav_yaw)]

				vel_cmd = TwistStamped()
				vel_cmd.header.stamp = rospy.Time.now()
				vel_cmd.header.frame_id = 'map'

				m, self.to_cmd, self.to_ai, _ = cm.step(ang_img, None)
				if flag_cmd == True:
					self.to_cmd = None

				print('{:.3f}\n'.format(self.uav_pose.pose.position.z))
				self.vel = [0, 0, 0, 0]
				self.vel[0] = m[2]*math.cos(self.aux_yaw) - m[3]*math.sin(self.aux_yaw)
				self.vel[1] = m[2]*math.sin(self.aux_yaw) + m[3]*math.cos(self.aux_yaw)
				self.vel[2] = m[1]
				self.vel[3] = m[0]

				vel_cmd.twist.linear.x = self.vel[0]
				vel_cmd.twist.linear.y = self.vel[1]
				vel_cmd.twist.linear.z = self.vel[2]
				vel_cmd.twist.angular.z = self.vel[3]

				self.velocity_pub.publish(vel_cmd)
				self.save_moment()

		pygame.quit()
		quit()

def main(stdscr):
	rospy.init_node('key_teleop')

	vel = MandoVel()
	vel.execute([10.0, 0.0, 0.0])


if __name__ == '__main__':
	curses.wrapper(main)