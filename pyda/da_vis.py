from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy
import os


def plot_gridded_data(lats, lons, data,title_str,
                      output_path, clim_range=None):
    llcrnrlon = lons[-1, 0]
    llcrnrlat = lats[0, -1]

    urcrnrlon = lons[0, -1]
    urcrnrlat = lats[-1, 0]
    
    lat_1 = numpy.mean(lats)
    lat_2 = numpy.mean(lats)

    lon_0 = numpy.mean(lons)

    m = Basemap(llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
                urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat,
                projection='lcc',
                lat_1=lat_1,lat_2=lat_2,lon_0=lon_0,
                resolution ='l')
    
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(numpy.arange(lats.min(),lats.max(),(lats.max()-lats.min())/5.0),labels=[1,1,0,0])
    m.drawmeridians(numpy.arange(lons.min(),lons.max(),(lons.max()-lons.min())/5.0),labels=[0,0,0,1])
    m.imshow(data)
    
    if clim_range:
        plt.clim(clim_range[0], clim_range[1])
    plt.title(title_str)
    plt.colorbar(orientation="horizontal",fraction=0.07)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def compare_background_and_analysis(background_data, analysis_data):
    model_fields = background_data.keys()
    
    model_lat = background_data['lat']
    model_lon = background_data['lon']
    
    for model_field in model_fields:
        if model_field == 'lat' or model_field == 'lon':
            continue
        cur_background_data = background_data[model_field]
        cur_analysis_data = analysis_data[model_field]
        
        clim_range = [min(cur_background_data.min(),
                          cur_analysis_data.min()),
                      max(cur_background_data.max(),
                          cur_analysis_data.max())]
        
        output_path = '/tmp/pyda'
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        plot_gridded_data(model_lat, model_lon, cur_background_data, 
                          'background: {}'.format(model_field),
                          os.path.join(output_path, 'background_{}.png'.format(model_field)),
                          clim_range=clim_range)

        plot_gridded_data(model_lat, model_lon, cur_analysis_data, 
                          'analysis: {}'.format(model_field),
                          os.path.join(output_path, 'analysis_{}.png'.format(model_field)),
                          clim_range=clim_range)
        
        plot_gridded_data(model_lat, model_lon, cur_analysis_data - cur_background_data, 
                          'analysis-background: {}'.format(model_field),
                          os.path.join(output_path, 'diff_{}.png'.format(model_field)))    
    
        