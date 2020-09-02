def main():
    path='/home/nicolaslam/data_compression/data/oppi_run1_cyc2038_raw.lh5'
    with h5py.File(path,'r') as hf:
        r,c = hf['ORSIS3302DecoderForEnergy/raw/waveform/values'].shape
        nda = hf['ORSIS3302DecoderForEnergy/raw/waveform/values'][()]

        flattened_data = np.empty(r*c, dtype = np.ushort)
        cumulative_length = np.empty(r+1, dtype=np.intc)

        print('Start compression')
        tic = time.time()
        length = nda_to_vect(nda, flattened_data, cumulative_length)
        flattened_data = np.resize(flattened_data,length)


        print('Cumulative_length: ',cumulative_length)
        print('flattened_data: ', flattened_data)
        print('Done!')
        elapsedTime = time.time()-tic
        inputsize = sys.getsizeof(nda)*1e-6
        print('Compression rate: %.2f Mb per sec'%(inputsize/elapsedTime))

        arr = empty(flattened_data, cumulative_length)
        tic = time.time()
        print("start decompression")
        vect_to_nda(flattened_data, cumulative_length, arr)
        print("Done!")
        print('%.2f sec' %(time.time() - tic))

         
        print(arr.all() == nda.all())

if __name__ == "__main__":
    main()
