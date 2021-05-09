import glob
import os

from pixellib.semantic import semantic_segmentation

import cv2 as cv
import numpy as np
import time


# Variables globales
n_mem = 5  # Número de frames de memoria
pinf_mem = []  # Array para almacenar puntos medios del segmento inferior
psup_mem = []  # Array para almacenar puntos medios del segmento inferior


# Rutina para segmentación semántica de imágenes en la carpeta 'img'
def imageSemanticSegmentation():
    # Carga del modelo para segmentación semántica
    pixelLib = semantic_segmentation()
    pixelLib.load_ade20k_model("deeplabv3_xception65_ade20k.h5")

    # Variables iniciales
    n = 0  # Número de imágenes segmentadas
    dur = 0  # Variable para el cálculo de tiempos

    for file in glob.glob("../img/*.jpg"):
        filename = os.path.splitext(os.path.basename(file))[0]
        if not filename.endswith("_seg") and not os.path.isfile("../img/" + filename + "_seg.jpg"):
            # Tiempo inicial
            t0 = time.time()

            # Segmentación de la imagen
            pixelLib.segmentAsAde20k(file, output_image_name="../img/" + filename + "_seg.jpg")

            # Cálculo de tiempos
            t1 = time.time()
            dur = dur + (t1 - t0)

            n = n + 1

    # Rendimiento de la segmentación de las imágenes
    if n != 0:
        print("La segmentación de " + str(n) + " imágenes ha durado " + str(round(dur, 2)) + " segundos.")
        print("Tiempo medio de segmentación por imagen: " + str(round(dur / n, 2)) + " segundos.")
    else:
        print("No se ha realizado la segmentación semántica de ninguna imagen.")


# Rutina para segmentación semántica de un video de la carpeta 'video'
def videoSemanticSegmentation(videoName):
    # Carga del modelo para segmentación semántica
    pixelLib = semantic_segmentation()
    pixelLib.load_ade20k_model("deeplabv3_xception65_ade20k.h5")

    # Variables iniciales
    video = None
    n = 0
    fps = 0
    dur = 0

    if not os.path.isfile("../video/" + videoName + "_seg.mp4"):
        video = cv.VideoCapture("../video/" + videoName + ".mp4")
        n = int(video.get(cv.CAP_PROP_FRAME_COUNT))  # Número de frames
        fps = int(video.get(cv.CAP_PROP_FPS))  # Frames Per Second
        video.release()

        # Tiempo inicial
        t0 = time.time()

        # Segmentación del video
        pixelLib.process_video_ade20k("../video/" + videoName + ".mp4", frames_per_second=fps,
                                      output_video_name="../video/" + videoName + "_seg.mp4")

        # Cálculo de tiempos
        t1 = time.time()  # Tiempo final
        dur = t1 - t0

    # Rendimiento de la segmentación del video
    if n != 0:
        print("La segmentación de " + str(n) + " frames ha durado " + str(round(dur, 2)) + " segundos.")
        print("Tiempo medio de segmentación por imagen: " + str(round(dur / n, 2)) + " segundos.")
    else:
        print("No se ha realizado la segmentación semántica del video.")


# Rutina para reescalado de imagen
def rescale(image):
    maxSize = 600
    width = image.shape[1]
    height = image.shape[0]

    if width > height:
        if width > maxSize:
            scale_percent = maxSize / width
            newHeight = int(height * scale_percent)

            dim = (maxSize, newHeight)

            resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)

            return resized
        else:
            return image
    else:
        if height > maxSize:
            scale_percent = maxSize / height
            newWidth = int(width * scale_percent)

            dim = (newWidth, maxSize)

            resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)

            return resized
        else:
            return image


# Rutina para extracción del suelo por color
def extractFloor(image):
    # Definicion del rango de color (en torno al [50, 50, 80], marron)
    lower_range = np.array([45, 45, 75])
    upper_range = np.array([55, 55, 85])

    floor = cv.inRange(image, lower_range, upper_range)

    return floor


# Rutina para limpieza de memoria
def limpiaMemoria():
    global pinf_mem, psup_mem
    pinf_mem = []
    psup_mem = []


# Rutina para cálculo de puntos medios de los segmentos superior e inferior
def midPoints(array, img_seg, img_floor):
    minY = None
    minIndex = None
    maxY = None
    maxIndex = None
    points = np.empty([2, 2], dtype=int)
    size = len(array)

    # Búsqueda de puntos con coordenada y mínima y máxima
    for i in range(0, size):
        y = array[i][0][1]

        if minY is None or y < minY:
            minY = y
            minIndex = i

        if maxY is None or y > maxY:
            maxY = y
            maxIndex = i

    # Variables para el cálculo del punto medio de ambos segmentos
    radioError = 10

    # --- Cálculo del punto medio (segmento superior) ---
    # Definición de nuevas variables
    minY_x = array[minIndex][0][0]
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    pendiente1 = 0
    pendiente2 = 0
    pAnt = None
    pSig = None

    # Casuística según índice del array para evitar accesos erróneos al array
    # Comparación de los puntos medios del segmento anterior y siguiente al punto de coordenada y mínima
    if minIndex == 0:
        pAnt = array[size - 1][0]
        pSig = array[minIndex + 1][0]
    elif minIndex == size - 1:
        pAnt = array[minIndex - 1][0]
        pSig = array[0][0]
    else:
        pAnt = array[minIndex - 1][0]
        pSig = array[minIndex + 1][0]

    x1 = (minY_x + pAnt[0]) / 2
    y1 = (minY + pAnt[1]) / 2
    x2 = (minY_x + pSig[0]) / 2
    y2 = (minY + pSig[1]) / 2

    difX1 = minY_x - x1
    difX2 = minY_x - x2

    if difX1 != 0 and difX2 != 0:
        pendiente1 = abs((minY - y1) / difX1)
        pendiente2 = abs((minY - y2) / difX2)

    if abs(pendiente1 - pendiente2) < 0.25 or (pendiente1 > 0.6 and pendiente2 > 0.6):
        # Mejora: Casuística. Forma de tipo triángulo isósceles
        x = minY_x
        y = minY
        points[0] = (x, y)
    elif abs(minY - y1) < radioError or abs(minY - y2) < radioError:
        # Mejora: Aparición de gran cantidad de puntos en el segmento superior
        minY_x_pointsCoords = []
        minY_pointsCoords = []

        for p in array:
            py = p[0][1]

            if (minY - radioError) <= py <= (minY + radioError):
                minY_x_pointsCoords.append(p[0][0])
                minY_pointsCoords.append(py)

        x = np.median(minY_x_pointsCoords)
        y = np.median(minY_pointsCoords)
        points[0] = (x, y)
    else:
        if y2 < y1:
            points[0] = (x2, y2)
        else:
            points[0] = (x1, y1)

    # --- Mejora para la desaparición lateral de parte del suelo (segmento inferior) ---
    # Definición de nuevas variables
    height, width = img_floor.shape[:2]
    lateralIzq_x = []
    lateralIzq_y = []
    lateralDer_x = []
    lateralDer_y = []
    lateralIzq_minIndex = -1
    lateralDer_minIndex = -1

    for p in array:
        px = p[0][0]

        if px == 0:
            lateralIzq_x.append(px)
            lateralIzq_y.append(p[0][1])
        elif px == width-1:
            lateralDer_x.append(px)
            lateralDer_y.append(p[0][1])

    if len(lateralIzq_x) != 0:
        lateralIzq_minIndex = np.argmin(lateralIzq_y)
    if len(lateralDer_x) != 0:
        lateralDer_minIndex = np.argmin(lateralDer_y)

    if lateralIzq_minIndex != -1 and lateralDer_minIndex == -1:  # Hay puntos por la izquierda
        point_x = lateralIzq_x[lateralIzq_minIndex]  # Coord. x del punto lateral izquierdo
        point_y = lateralIzq_y[lateralIzq_minIndex]  # Coord. y del punto lateral izquierdo
        cv.circle(img_seg, (point_x, point_y), 0, (0, 255, 0), 10)

        # Búsqueda del punto esquina inferior derecho (2 candidatos de máxima coord. Y)
        maxY_points_x = []
        maxY_points_y = []

        for p in array:
            py = p[0][1]

            if len(maxY_points_y) < 2:
                maxY_points_x.append(p[0][0])
                maxY_points_y.append(py)
            else:
                # Se reemplaza el punto de menor coord. y del array por el nuevo máximo
                yMinIndex = maxY_points_y.index(min(maxY_points_y))  # Índice del elemento de mínima coord. Y

                if py > maxY_points_y[yMinIndex]:
                    maxY_points_x[yMinIndex] = p[0][0]
                    maxY_points_y[yMinIndex] = py

        xMaxIndex = maxY_points_x.index(max(maxY_points_x))  # Índice del elemento de máxima coord. X
        esqInfDer_x = maxY_points_x[xMaxIndex]
        esqInfDer_y = maxY_points_y[xMaxIndex]
        cv.circle(img_seg, (esqInfDer_x, esqInfDer_y), 0, (0, 255, 0), 10)

        # Se obtiene el punto anterior (sentido horario) del punto lateral izquierdo
        antIndex = None
        i = 0

        for p in array:
            if p[0][0] == point_x and p[0][1] == point_y:
                antIndex = (i - 1) % len(array)
                break
            i = i + 1

        pAnt = array[antIndex]
        cv.circle(img_seg, (pAnt[0][0], pAnt[0][1]), 0, (0, 255, 0), 10)

        # Se calcula la pendiente entre el punto del lateral izquierdo y el punto anterior y se usa
        # para averiguar el punto "imaginario" de corte con el eje X
        difX = point_x - pAnt[0][0]

        if difX != 0:
            pendiente = (point_y - pAnt[0][1]) / difX

            if abs(pendiente) > 0.01:
                pcorte_x = (esqInfDer_y - point_y) / pendiente + point_x

                points[1] = ((pcorte_x + esqInfDer_x) / 2, esqInfDer_y)

    elif lateralIzq_minIndex == -1 and lateralDer_minIndex != -1:  # Hay puntos por la derecha
        point_x = lateralDer_x[lateralDer_minIndex]
        point_y = lateralDer_y[lateralDer_minIndex]
        cv.circle(img_seg, (point_x, point_y), 0, (0, 255, 0), 10)

        # Búsqueda del punto esquina inferior izquierdo (2 candidatos de máxima coord. Y)
        maxY_points_x = []
        maxY_points_y = []

        for p in array:
            py = p[0][1]

            if len(maxY_points_y) < 2:
                maxY_points_x.append(p[0][0])
                maxY_points_y.append(py)
            else:
                # Se reemplaza el punto de menor coord. y del array por el nuevo máximo
                yMinIndex = maxY_points_y.index(min(maxY_points_y))  # Índice del elemento de mínima coord. Y

                if py > maxY_points_y[yMinIndex]:
                    maxY_points_x[yMinIndex] = p[0][0]
                    maxY_points_y[yMinIndex] = py

        xMinIndex = maxY_points_x.index(min(maxY_points_x))  # Índice del elemento de mínima coord. X
        esqInfIzq_x = maxY_points_x[xMinIndex]
        esqInfIzq_y = maxY_points_y[xMinIndex]
        cv.circle(img_seg, (esqInfIzq_x, esqInfIzq_y), 0, (0, 255, 0), 10)

        # Se obtiene el punto siguiente (sentido antihorario) del punto lateral derecho
        sigIndex = None
        i = 0

        for p in array:
            if p[0][0] == point_x and p[0][1] == point_y:
                sigIndex = (i + 1) % len(array)
                break
            i = i + 1

        pSig = array[sigIndex]
        cv.circle(img_seg, (pSig[0][0], pSig[0][1]), 0, (0, 255, 0), 10)

        # Se calcula la pendiente entre el punto del lateral derecho y el punto siguiente y se usa
        # para averiguar el punto "imaginario" de corte con el eje X
        difX = point_x - pSig[0][0]

        if difX != 0:
            pendiente = (point_y - pSig[0][1]) / difX

            if abs(pendiente) > 0.01:
                pcorte_x = (esqInfIzq_y - point_y) / pendiente + point_x

                points[1] = ((pcorte_x + esqInfIzq_x) / 2, esqInfIzq_y)

    elif lateralIzq_minIndex != -1 and lateralDer_minIndex != -1:  # Hay puntos por ambos lados
        pointDer_x = lateralDer_x[lateralDer_minIndex]
        pointDer_y = lateralDer_y[lateralDer_minIndex]
        pointIzq_x = lateralIzq_x[lateralIzq_minIndex]
        pointIzq_y = lateralIzq_y[lateralIzq_minIndex]

        cv.circle(img_seg, (pointDer_x, pointDer_y), 0, (0, 255, 0), 10)
        cv.circle(img_seg, (pointIzq_x, pointIzq_y), 0, (0, 255, 0), 10)

        # Se obtiene el punto anterior (sentido horario) del punto lateral izquierdo
        antIndex = None
        i = 0

        for p in array:
            if p[0][0] == pointIzq_x and p[0][1] == pointIzq_y:
                antIndex = (i - 1) % len(array)
                break
            i = i + 1

        pAnt = array[antIndex]
        cv.circle(img_seg, (pAnt[0][0], pAnt[0][1]), 0, (0, 255, 0), 10)

        # Se obtiene el punto siguiente (sentido antihorario) del punto lateral derecho
        sigIndex = None
        i = 0

        for p in array:
            if p[0][0] == pointDer_x and p[0][1] == pointDer_y:
                sigIndex = (i + 1) % len(array)
                break
            i = i + 1

        pSig = array[sigIndex]
        cv.circle(img_seg, (pSig[0][0], pSig[0][1]), 0, (0, 255, 0), 10)

        # Se calcula la pendiente entre el punto del lateral izquierdo y el punto medio superior y se usa
        # para averiguar el punto "imaginario" de corte con el eje X
        difDerX = pointDer_x - pSig[0][0]
        difIzqX = pointIzq_x - pAnt[0][0]

        if difDerX != 0 and difIzqX != 0:
            pendienteDer = (pointDer_y - pSig[0][1]) / difDerX
            pendienteIzq = (pointIzq_y - pAnt[0][1]) / difIzqX

            if abs(pendienteDer) > 0.01 and abs(pendienteIzq) > 0.01:
                maxLateralDerY = max(lateralDer_y)
                maxLateralIzqY = max(lateralIzq_y)

                pcorteDer_x = (maxLateralDerY - pointDer_y) / pendienteDer + pointDer_x
                pcorteIzq_x = (maxLateralIzqY - pointIzq_y) / pendienteIzq + pointIzq_x

                points[1] = ((pcorteDer_x + pcorteIzq_x) / 2, (maxLateralDerY + maxLateralIzqY) / 2)

    else:  # No hay puntos laterales
        # --- Cálculo del punto medio (segmento inferior) ---
        # Definición de nuevas variables
        maxY_x = array[maxIndex][0][0]
        x1 = 0
        y1 = 0
        x2 = 0
        y2 = 0
        pAnt = None
        pSig = None

        # Casuística según índice del array para evitar accesos erróneos al array
        # Comparación de los puntos medios del segmento anterior y siguiente al punto de coordenada y máxima
        if maxIndex == 0:
            pAnt = array[size - 1][0]
            pSig = array[maxIndex + 1][0]
        elif maxIndex == size - 1:
            pAnt = array[maxIndex - 1][0]
            pSig = array[0][0]
        else:
            pAnt = array[maxIndex - 1][0]
            pSig = array[maxIndex + 1][0]

        x1 = (maxY_x + pAnt[0]) / 2
        y1 = (maxY + pAnt[1]) / 2
        x2 = (maxY_x + pSig[0]) / 2
        y2 = (maxY + pSig[1]) / 2

        if abs(y2 - y1) < 5:
            # Mejora: Punto medio del segmento inferior es el punto medio entre pAnt y pSig
            # Soluciona defectos en la parte inferior del polígono envolvente
            x = (pAnt[0] + pSig[0]) / 2
            y = (pAnt[1] + pSig[1]) / 2
            points[1] = (x, y)
        else:
            if y2 > y1:
                points[1] = (x2, y2)
            else:
                points[1] = (x1, y1)

    return points


# Rutina para extracción de suelo y procesamiento
def floorAndContours(inputFrame):
    global pinf_mem, psup_mem

    # Copia local de la imagen
    frame_seg = inputFrame

    # Extracción del suelo a partir de la imagen segmentada
    frame_floor = extractFloor(frame_seg)  # Haciendo uso de un rango de color marron

    kernel_close = np.ones(5, np.uint8)
    kernel_open = np.ones(3, np.uint8)
    frame_floor = cv.morphologyEx(frame_floor, cv.MORPH_CLOSE,
                                  kernel_close)  # Operación morfológica de cierre para tapar agujeros
    frame_floor = cv.morphologyEx(frame_floor, cv.MORPH_OPEN,
                                  kernel_open)  # Operación morfológica de apertura para limpiar defectos

    # Aproximación poligonal del contorno del suelo extraído
    contours, hierarchy = cv.findContours(frame_floor, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    contoursSize = len(contours)

    if contoursSize != 0:
        accuracy = 0.01 * cv.arcLength(contours[contoursSize-1], True)
        approx = cv.approxPolyDP(contours[contoursSize-1], accuracy, True)

        # Se dibujan los puntos medios de los segmentos superior e inferior del pasillo
        if len(approx) >= 3:
            cv.drawContours(frame_seg, [approx], 0, (0, 0, 255), 2)

            points = midPoints(approx, frame_seg, frame_floor)
            p1 = (points[0][0], points[0][1])  # Punto medio del segmento superior
            p2 = (points[1][0], points[1][1])  # Punto medio del segmento inferior

            # Cálculo de la media de la dirección (n_mem frames de memoria)
            # psup -> Puntos medio del segmento superior (n_mem frames)
            # pinf -> Puntos medio del segmento inferior (n_mem frames)
            psup_mem_len = len(psup_mem)

            if psup_mem_len < n_mem:
                # if abs(p2[1] - p1[1]) > 30:
                psup_mem.append(p1)
                pinf_mem.append(p2)

                psup_mem_len = psup_mem_len + 1
            else:
                # Desplazamiento a la izquierda e inserción del nuevo elemento
                psup_ant = psup_mem[1:n_mem]
                pinf_ant = pinf_mem[1:n_mem]

                psup_ant.append(p1)
                pinf_ant.append(p2)

                psup_mem = psup_ant
                pinf_mem = pinf_ant

            # Media de los puntos medios
            psup_sum = np.sum(psup_mem, axis=0)
            pinf_sum = np.sum(pinf_mem, axis=0)

            p1_x = int(psup_sum[0] / psup_mem_len)
            p1_y = int(psup_sum[1] / psup_mem_len)
            p2_x = int(pinf_sum[0] / psup_mem_len)
            p2_y = int(pinf_sum[1] / psup_mem_len)

            p1 = (p1_x, p1_y)
            p2 = (p2_x, p2_y)

            cv.circle(frame_seg, p1, 0, (255, 0, 0), 10)
            cv.circle(frame_seg, p2, 0, (255, 0, 0), 10)
            cv.arrowedLine(frame_seg, p2, p1, (0, 255, 0), 1)

    return frame_seg, frame_floor


# Código para imágenes
def imageProcessing(imageName):
    # Segmentación semántica de imágenes (cuando no se ha hecho previamente)
    imageSemanticSegmentation()

    # Procesamiento de la imagen especificada
    if os.path.isfile("../img/" + imageName + ".jpg"):
        org = cv.imread("../img/" + imageName + ".jpg")
        img_seg = cv.imread("../img/" + imageName + "_seg.jpg")

        # Reescalamiento de imágenes
        org = rescale(org)
        img_seg = rescale(img_seg)

        # Extracción de suelo y procesamiento
        img_seg, img_floor = floorAndContours(img_seg)

        # Comparación de resultados gráficos
        cv.imshow('Org', org)
        cv.imshow('Floor', img_floor)
        cv.imshow('Contours', img_seg)
        cv.imshow('Comparison', cv.addWeighted(org, 0.5, img_seg, 0.5, 0))

        while True:
            if cv.waitKey(1) & 0xFF == ord('q'):
                cv.destroyAllWindows()
                break
    else:
        print("No se ha podido encontrar la imagen en la ruta definida.")


# Código para videos
def videoProcessing(videoName):
    if os.path.isfile("../video/" + videoName + ".mp4"):
        videoSemanticSegmentation(videoName)
        video_org = cv.VideoCapture("../video/" + videoName + ".mp4")
        video_seg = cv.VideoCapture("../video/" + videoName + "_seg.mp4")

        # Velocidad de reproducción del video
        playbackSpeed = float(input("Velocidad de reproducción del video (entre 0.1 y 1): "))
        while playbackSpeed < 0.1 or playbackSpeed > 1:
            playbackSpeed = float(input("Velocidad inválida. Vuelva a insertar una velocidad (entre 0.1 y 1): "))

        while video_seg.isOpened():
            ret_org, frame_org = video_org.read()
            ret_seg, frame_seg = video_seg.read()

            if frame_seg is not None:
                # Reescalado de frames
                frame_org = rescale(frame_org)
                frame_seg = rescale(frame_seg)

                # Extracción de suelo y procesamiento
                frame_seg, frame_floor = floorAndContours(frame_seg)

                # Muestra de los resultados
                cv.imshow('Org', frame_org)
                cv.imshow('Floor', frame_floor)
                cv.imshow('Contours', frame_seg)
                cv.imshow('Comparison', cv.addWeighted(frame_org, 0.5, frame_seg, 0.5, 0))

                if playbackSpeed != 1:
                    time.sleep(0.1/playbackSpeed)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    video_org.release()
                    video_seg.release()
                    break
            else:
                video_org.release()
                video_seg.release()
                break

        cv.destroyAllWindows()
    else:
        print("No se ha podido encontrar el video en la ruta definida.")


# Código para cámara
def camaraProcessing(intCamara):
    # Carga del modelo para segmentación semántica
    pixelLib = semantic_segmentation()
    pixelLib.load_ade20k_model("deeplabv3_xception65_ade20k.h5")

    # Apertura de la cámara
    camara = cv.VideoCapture(intCamara)

    # Variables iniciales
    n = 0
    t0 = time.time()

    while camara.isOpened():
        ret, frame_org = camara.read()

        if frame_org is not None:
            n = n + 1

            # Segmentación semántica del frame
            _, frame_seg = pixelLib.segmentAsAde20k(frame_org, process_frame=True)

            # Reescalado de frames
            frame_org = rescale(frame_org)
            frame_seg = rescale(frame_seg)

            # Extracción de suelo y procesamiento
            frame_seg, frame_floor = floorAndContours(frame_seg)

            # Muestra de los resultados
            cv.imshow('Org', frame_org)
            cv.imshow('Floor', frame_floor)
            cv.imshow('Contours', frame_seg)
            cv.imshow('Comparison', cv.addWeighted(frame_org, 0.5, frame_seg, 0.5, 0))

            # Cálculo de tiempos
            t1 = time.time()
            dur = t1 - t0

            if dur >= 5:
                fps = round(n / dur, 2)
                print("Tasa de FPS en 5 segundos: " + str(fps) + " frames por segundo.")

                # Reinicio de variables para cálculo del tiempo
                n = 0
                t0 = time.time()

            if cv.waitKey(1) & 0xFF == ord('q'):
                camara.release()
                break
        else:
            camara.release()
            break

    cv.destroyAllWindows()


if __name__ == '__main__':
    # Menú de selección de opciones disponibles
    print("-----------------------------------------------------------------------")
    print("[1] Segmentación + Post-procesamiento de imágenes en la carpeta 'img'.")
    print("[2] Segmentación + Post-procesamiento de un video en la carpeta 'video'.")
    print("[3] Segmentación + Post-procesamiento en una cámara.")
    print("[0] Salir del programa")
    print("-----------------------------------------------------------------------")

    option = int(input("\nIntroduce una de las opciones disponibles: "))

    while option != 0:
        if option == 1:
            nombre = input("Introduce el nombre de una imagen (por ejemplo, 'corridor1', sin extensión .jpg y sin "
                           "apóstrofes): ")
            imageProcessing(nombre)
        elif option == 2:
            nombre = input("Introduce el nombre de un video (por ejemplo, 'corridor1', sin extensión .mp4 y sin "
                           "apóstrofes): ")
            videoProcessing(nombre)
        elif option == 3:
            cam = int(input("Introduce el número asociado a la cámara objetivo (por defecto, 0): "))
            camaraProcessing(cam)
        else:
            print("Opción no válida.")

        limpiaMemoria()
        option = int(input("\nIntroduce una nueva opción: "))

    print("\nSaliendo del programa...")
