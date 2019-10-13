import cv2
import time
import numpy as np
from matplotlib import pyplot as plt


DELAY = 0.02
USE_CAM = 1
IS_FOUND = 0

MORPH = 7
CANNY = 250

_width  = 840.0
_height = 588.0
_margin = 0.0
##################

if USE_CAM: video_capture = cv2.VideoCapture(0)

corners = np.array(
	[
		[[  		_margin, _margin 			]],
		[[ 			_margin, _height + _margin  ]],
		[[ _width + _margin, _height + _margin  ]],
		[[ _width + _margin, _margin 			]],
	]
)

pts_dst = np.array( corners, np.float32 )

while True :

	if USE_CAM :
		ret, rgb = video_capture.read()
	else :
		ret = 1
		rgb = cv2.imread( "opencv.jpg", 1 )

	if ( ret ):

		gray = cv2.cvtColor( rgb, cv2.COLOR_BGR2GRAY )

		gray = cv2.bilateralFilter( gray, 1, 10, 120 )

		edges  = cv2.Canny( gray, 10, CANNY )

		kernel = cv2.getStructuringElement( cv2.MORPH_RECT, ( MORPH, MORPH ) )

		closed = cv2.morphologyEx( edges, cv2.MORPH_CLOSE, kernel )

		contours, h = cv2.findContours( closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

		for cont in contours:

			# Küçük alanları pass geç
			if cv2.contourArea( cont ) > 5000 :

				arc_len = cv2.arcLength( cont, True )

				approx = cv2.approxPolyDP( cont, 0.1 * arc_len, True )

				if ( len( approx ) == 4 ):
					IS_FOUND = 1


					pts_src = np.array( approx, np.float32 )

					h, status = cv2.findHomography( pts_src, pts_dst )
					out = cv2.warpPerspective( rgb, h, ( int( _width + _margin* 2  ), int( _height + _margin* 2 ) ) )

					cv2.drawContours( rgb, [approx], -1, ( 255, 0, 0 ), 2 )


				else : pass

		#cv2.imshow( 'closed', closed )
		#cv2.imshow( 'gray', gray )
		cv2.namedWindow( 'edges', cv2.WINDOW_AUTOSIZE)
		cv2.imshow( 'edges', edges )

		cv2.namedWindow( 'rgb', cv2.WINDOW_AUTOSIZE )
		cv2.imshow( 'rgb', rgb )

		if IS_FOUND :
			cv2.namedWindow( 'out', cv2.WINDOW_AUTOSIZE )
			cv2.imshow( 'out', out )
			cv2.imwrite('out.jpg',out)

		if cv2.waitKey(27) & 0xFF == ord('q') :
			break

		if cv2.waitKey(99) & 0xFF == ord('c') :
			current = str(time.time())
			cv2.imwrite('ocvi_' + current + '_edges.jpg', edges)
			cv2.imwrite('ocvi_' + current + '_gray.jpg', gray)
			cv2.imwrite('ocvi_' + current + '_org.jpg', rgb)
			print("Pictures saved")
			son = cv2.imread('out.jpg')

			img_gray = cv2.cvtColor(son, cv2.COLOR_BGR2GRAY)
			img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
			cv2.imshow ('gray', img_gray)

			template = cv2.imread('isaret.jpg', 0)
			template = cv2.GaussianBlur(template, (5, 5), 0)

			w, h = template.shape[::-1]

			res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

			threshold = 0.9

			loc = np.where(res >= threshold)

			for pt in zip(*loc[::-1]):
				cv2.rectangle(son, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 0)

			#son[165:210,5:10]=[255,255,255]



			#cv2.imshow('beyaz',son)



			cv2.imwrite('son.png', son)
			cv2.imshow('son', son)
			cv2.imread('son.jpg', cv2.IMREAD_COLOR)





		time.sleep(DELAY)

	else :
		print ("Stopped")
		break


if USE_CAM : video_capture.release()


cv2.destroyAllWindows()

# end print(pt)
# 				for i in range(100, 500):
# 					for j in range(1, 300):
# 						if pt(j, i) == pt[1:100, 10:120]:
# 							print("A")
