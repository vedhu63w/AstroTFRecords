
# Takes fits file as input and outputs a tfrecord
import argparse
from astropy.io             import fits
from glob                   import glob as glob_g
from numpy                  import zeros, float32 as np_float32
from os                     import makedirs as os_makedir
from os.path                import exists as os_exists, join as os_join

import sys
import tensorflow           as tf


img_dim = 21
num_img = 5
# dir_nm = "/home/patel.3140/ASSASIN/Astro_Code/TestInput/Test_astro/Deep_22_15k/deep/deepl3"
# out_dir = "/home/patel.3140/ASSASIN/Astro_Code/TestInput/Test_astro/Deep_22_15k/deep/TFRecord"


def writable(dir_nm, fl_nm):
    if not os_exists(dir_nm):
        os_makedir(dir_nm)
    return os_join(dir_nm, fl_nm)


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def convert_to(images, labels, name, num_examples):
  """Converts a dataset to tfrecords."""

  if images.shape[0] != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))
  rows, cols, depth = images.shape[1], images.shape[2], images.shape[3]

  filename = name + '.tfrecords'
  with tf.python_io.TFRecordWriter(filename) as writer:
    for index in range(num_examples):
      image_raw = images[index].tostring()
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'height': _int64_feature(rows),
                  'width': _int64_feature(cols),
                  'depth': _int64_feature(depth),
                  'label': _int64_feature(int(labels[index])),
                  'image_raw': _bytes_feature(image_raw)
              }))
      writer.write(example.SerializeToString())



def main(opts):
    dir_nm, out_dir = opts.inp_fits_dir[0], opts.out_dir[0]
    list_files = glob_g(dir_nm+'/*.fits')
    tot_images = len(list_files)
    images = zeros((tot_images, num_img, img_dim, img_dim), dtype=np_float32)
    fl_nm = []
    RFC = zeros(tot_images)   #Random Forest Score
    TrueClass = zeros(tot_images, dtype=int)    #True Labels
    Rtype = zeros(7)   # Maintain the count of the classes
    print ("Total Fits Files: %d" % len(list_files))
    i = 0
    split = 0.9
    # Use this if train set and test set are needed for the inp_dir and replace list_files in next line by train_set/test_set 
    # train_set = list_files[:int(len(list_files)*split)]
    # test_set = list_files[int(len(list_files)*split):]
    for file in list_files:
        hdulist = fits.open(file)
        try:
            Rtype[hdulist[1].header['RTYPE']] += 1
            # Ignoring class 3 and 4 for now
            if hdulist[1].header['RTYPE'] == 3 or hdulist[1].header['RTYPE'] == 4: continue
            for j in range(1, num_img+1):
                images[i][j-1] = hdulist[j].data
            RFC[i] = hdulist[1].header['RFCVAL']
            # Making it binary class
            # Should not do it like this 
            TrueClass[i] = ( int(hdulist[1].header['RTYPE'] / 4) ) ^ 1 
            hdulist.close()
            fl_nm.append(""+file.split('/')[-1])
            i+=1
        except Exception as e:
            # Not Using python logger currently 
            with open (writable(out_dir, "Failed_Fits"), 'a+') as fl_out:
                fl_out.write(file.split('/')[-1]+ " : ")
                fl_out.write(str(e)+"\n")
            continue
    convert_to(images[:i], TrueClass[:i], writable(out_dir, opts.out_fl_nm[0]), i)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_fits_dir', nargs=1, help='The location of the fitsfiles')
    parser.add_argument('--out_fl_nm', nargs=1, help='Name for the output tfrecords file')
    parser.add_argument('--out_dir', nargs=1,  default='./', help='Output Location')
    opts = parser.parse_args()
    main(opts)
