#include <bits/stdc++.h>
#include <caffe/caffe.hpp>

using namespace std;

class Classifier{
    public:
    Classifier(const string& model_file,
             const string& trained_file);

    vector<vector<vector<int>>> eval(vector<vector<vector<int>>> &im);

    private:
    caffe::shared_ptr<Net<float>> net_;
};

Classifier::Classifier(const string& model_file,
             const string& trained_file)
            {
                net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
                net_->CopyTrainedLayersFrom(trained_file);

                caffe::Blob<float>* input_layer = net_->input_blobs()[0];
                input_layer->set_cpu_data(your_image_data_as_pointer_to_float);
                num_channels_ = input_layer->channels();

                caffe::Blob<float>* output_layer = net_->output_blobs()[0];
            }

vector<vector<vector<int>>> eval(vector<vector<vector<int>>> &im)
{
    int nh=224;
    int nw=224;

    vector<double> img_mean {103.94, 116.78, 123.68};

    int h=im.size();
    int w=im[0].size();

    cout<<"sizes = "<<im.size()<<" "<<im[0].size()<<" "<<im[0][0].size()<<"\n";

    vector<vector<vector<int>>> output;

    if(h<w)
    {
        output.resize(h,vector<vector<int> >(h,vector<int>(3)));
        int off=(w-h)/2;
        for(int i=0;i<h;i++){
            for(int j=off;j<off+h;j++){
                for(int k=0;k<3;k++){
                    output[i][j-off][k]=im[i][j][k];
                }
            }
        }
    }
    else
    {
        output.resize(h,vector<vector<int> >(w,vector<int>(3)));
        int off=(h-w)/2;
        for(int i=off;i<h;i++){
            for(int j=0;j<w;j++){
                for(int k=0;k<3;k++){
                    output[i-off][j][k]=im[i][j][k];
                    cout<<"check\n";
                }
            }
        }
    }

    //RESIZE

    for(int i=0;i<output.size();i++)
    {
        output[i].resize(nw,vector<int>(3,0));
    }
    output.resize(nh,vector<vector<int>>(nw,vector<int>(3,0)));

    //SET TRANSPOSE

    vector<vector<vector<int>>>  out(3,vector<vector<int>> (output.size(),vector<int> (output[0].size(),0)));
    for(int i=0;i<3;i++){
        for(int j=0;j<output.size();j++){
            for(int k=0;k<output[0].size();k++){
                out[i][j][k]=(output[j][k][i]);
            }
        }
    }

    //CHANNEL SWAP

    vector<vector<vector<int>>>  o(3,vector<vector<int>> (output.size(),vector<int> (output[0].size(),0)));
    o[0]=out[2];
    o[1]=out[1];
    o[2]=out[0];

    //SET_RAW_SCALE

    for(int i=0;i<o.size();i++){
        for(int j=0;j<o[0].size();j++){
            for(int k=0;k<o[0][0].size();k++){
                o[i][j][k]*=255;
            }
        }
    }

    //SET MEAN

    vector<vector<vector<double>>> mean3(3,vector<vector<double>> (1,vector<double> (1,0)));
    for(int i=0;i<3;i++){
        for(int j=0;j<1;j++){
            for(int k=0;k<1;k++){
                mean3[i][j][k]=img_mean[i];
            }
        }
    }

    //SET INPUT SCALE

    double input_scale=0.017;

    

    caffe::Blob<float>* input_layer=net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                       nh, nw);

    net_->Reshape();

    return o;

}

signed main()
{
    // ios::sync_with_stdio(false);
    // cin.tie(0); cout.tie(0);

    vector<vector<vector<int>>> input(1,vector<vector<int>> (2,vector<int> (3,0)));

    for(int i=0;i<1;i++)
    {
        for(int j=0;j<2;j++)
        {
            for(int k=0;k<3;k++)
            {
                input[i][j][k]=i+j+k;
            }
        }
    }

    vector<vector<vector<int>>> output=eval(input);

    for(int k=0;k<output.size();k++)
    {
        for(int i=0;i<output[0].size();i++)
        {
            for(int j=0;j<output[0][0].size();j++)
            {
                cout<<output[k][i][j]<<" ";
            }
            cout<<"\n";
        }
    }
   
}