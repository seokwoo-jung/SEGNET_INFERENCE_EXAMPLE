TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    c_segmentator.cpp

# Opencv libs --------------------------------------
INCLUDEPATH += /usr/local/include

LIBS += -L/usr/local/lib    \
    -lopencv_core \
    -lopencv_highgui \
    -lopencv_imgproc \
    -lopencv_imgcodecs \
    -lopencv_video \
    -lopencv_videoio
# --------------------------------------------------

# caffe libs ---------------------------------------
INCLUDEPATH += $(HOME)/caffe-merge/include   \
    /usr/local/cuda/include

LIBS += -L$(HOME)/caffe-merge/build/lib \
    -lcaffe
# --------------------------------------------------

# etc ----------------------------------------------
LIBS += -L/usr/lib/x86_64-linux-gnu \
    -lboost_system  \
    -lglog \
    -lgflags
# --------------------------------------------------

HEADERS += \
    c_segmentator.h
