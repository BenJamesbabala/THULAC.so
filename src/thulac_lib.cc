#include "thulac_base.h"
#include "preprocess.h"
#include "postprocess.h"
#include "punctuation.h"
#include "cb_tagging_decoder.h"
#include "chinese_charset.h"
#include "thulac.h"
#include "filter.h"
#include "timeword.h"
#include "verbword.h"
#include "negword.h"
#include "wb_extended_features.h"
#include "wb_lattice.h"
#include "bigram_model.h"
#include <sstream>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cstring>
using namespace thulac;
namespace {
	bool checkfile(const char* filename){
		std::fstream infile;
		infile.open(filename, std::ios::in);
		if(!infile){
			return false;
		} else {
			infile.close();
			return true;
		}
	}
}

namespace {
	char* user_specified_dict_name=NULL;
	char* model_path_char=NULL;
	Character separator = '_';
	bool useT2S = false;
	bool seg_only = false;
	bool useFilter = false;
	bool use_second = false;
	TaggingDecoder* cws_decoder=NULL;
	permm::Model* cws_model = NULL;
	DAT* cws_dat = NULL;
	char** cws_label_info = NULL;
	int** cws_pocs_to_tags = NULL;
	int result_size = 0;

	TaggingDecoder* tagging_decoder = NULL;
	permm::Model* tagging_model = NULL;
	DAT* tagging_dat = NULL;
	char** tagging_label_info = NULL;
	int** tagging_pocs_to_tags = NULL;
	LatticeFeature* lf = NULL;
	DAT* sogout = NULL;
	std::vector<std::string> n_gram_model;
	std::vector<std::string> dictionaries;
	hypergraph::Decoder<int,LatticeEdge> decoder;
	Preprocesser* preprocesser = NULL;

	Postprocesser* ns_dict = NULL;
	Postprocesser* idiom_dict = NULL;
	Postprocesser* nz_dict = NULL;
	Postprocesser* ni_dict = NULL;
	Postprocesser* noun_dict = NULL;
	Postprocesser* adj_dict = NULL;
	Postprocesser* verb_dict = NULL;
	// Postprocesser* vm_dict = NULL;
	Postprocesser* y_dict = NULL;
	Postprocesser* user_dict = NULL;

	Punctuation* punctuation = NULL;

	NegWord* negword = NULL;
	TimeWord* timeword = NULL;
	VerbWord* verbword = NULL;

	Filter* filter = NULL;
	char *result = NULL;
}

extern "C" int init(const char * model, const char* dict = NULL, int ret_size = 1024 * 1024 * 16,int t2s = 0, int just_seg=0) {
	if (ret_size < 4) {
		return int(false);
	}
	result_size = ret_size;
	user_specified_dict_name = const_cast<char *>(dict);
	model_path_char = const_cast<char *>(model);
	separator = '_';
	useT2S = bool(t2s);
	seg_only = bool(just_seg);
	std::string prefix;
	if(model_path_char != NULL){
		prefix = model_path_char;
		if(*prefix.rbegin() != '/'){
			prefix += "/";
		}
	}else{
		prefix = "models/";
	}

	cws_decoder=new TaggingDecoder();
	if(seg_only){
		cws_decoder->threshold=0;
	}else{
		cws_decoder->threshold=15000;
	}
	cws_model = new permm::Model((prefix+"cws_model.bin").c_str());
	cws_dat = new DAT((prefix+"cws_dat.bin").c_str());
	cws_label_info = new char*[cws_model->l_size];
	cws_pocs_to_tags = new int*[16];
	get_label_info((prefix+"cws_label.txt").c_str(), cws_label_info, cws_pocs_to_tags);
	cws_decoder->init(cws_model, cws_dat, cws_label_info, cws_pocs_to_tags);
	cws_decoder->set_label_trans();
	if(!seg_only){
		tagging_decoder = new TaggingDecoder();
		tagging_decoder->separator = separator;
		if(use_second){
			tagging_decoder->threshold = 10000;
		}else{
			tagging_decoder->threshold = 0;
		}
		tagging_model = new permm::Model((prefix+"model_c_model.bin").c_str());
		tagging_dat = new DAT((prefix+"model_c_dat.bin").c_str());
		tagging_label_info = new char*[tagging_model->l_size];
		tagging_pocs_to_tags = new int*[16];
	
		get_label_info((prefix+"model_c_label.txt").c_str(), tagging_label_info, tagging_pocs_to_tags);
		tagging_decoder->init(tagging_model, tagging_dat, tagging_label_info, tagging_pocs_to_tags);
		tagging_decoder->set_label_trans();
	}
	lf = new LatticeFeature();
	sogout=new DAT((prefix+"sgT.dat").c_str());
	lf->node_features.push_back(new SogouTFeature(sogout));
	dictionaries.push_back(prefix+"sgW.dat");
	for(int i=0;i<dictionaries.size();++i){
		lf->node_features.push_back(new DictNodeFeature(new DAT(dictionaries[i].c_str())));
	}
	lf->filename=prefix+"model_w";
	lf->load();
	decoder.features.push_back(lf);
	preprocesser = new Preprocesser();
	preprocesser->setT2SMap((prefix+"t2s.dat").c_str());

	ns_dict = new Postprocesser((prefix+"ns.dat").c_str(), "ns", false);
	idiom_dict = new Postprocesser((prefix+"idiom.dat").c_str(), "i", false);
	nz_dict = new Postprocesser((prefix+"nz.dat").c_str(), "nz", false);
	ni_dict = new Postprocesser((prefix+"ni.dat").c_str(), "ni", false);
	noun_dict = new Postprocesser((prefix+"noun.dat").c_str(), "n", false);
	adj_dict = new Postprocesser((prefix+"adj.dat").c_str(), "a", false);
	verb_dict = new Postprocesser((prefix+"verb.dat").c_str(), "v", false);
	// vm_dict = new Postprocesser((prefix+"vm.dat").c_str(), "vm", false);
	y_dict = new Postprocesser((prefix+"y.dat").c_str(), "y", false);

	if(user_specified_dict_name){
		user_dict = new Postprocesser(user_specified_dict_name, "uw", true);
	}

	punctuation = new Punctuation((prefix+"singlepun.dat").c_str());

	negword = new NegWord((prefix+"neg.dat").c_str());
	timeword = new TimeWord();
	verbword = new VerbWord((prefix+"vM.dat").c_str(), (prefix+"vD.dat").c_str());

	filter = NULL;
	if(useFilter){
		filter = new Filter((prefix+"xu.dat").c_str(), (prefix+"time.dat").c_str());
	}
	return int(true);
}

extern "C" void deinit() {
	delete preprocesser;
	delete ns_dict;
	delete idiom_dict;
	delete nz_dict;
	delete ni_dict;
	delete noun_dict;
	delete adj_dict;
	delete verb_dict;
	// delete vm_dict;
	delete y_dict;
	if(user_dict != NULL){
		delete user_dict;
	}

	delete negword;
	delete timeword;
	delete verbword;
	delete punctuation;
	if(useFilter){
		delete filter;
	}

	delete lf;

	delete cws_decoder;
	if(cws_model != NULL){
		for(int i = 0; i < cws_model->l_size; i ++){
			if(cws_label_info) delete[](cws_label_info[i]);
		}
	}
	delete[] cws_label_info;

	if(cws_pocs_to_tags){
		for(int i = 1; i < 16; i ++){
			delete[] cws_pocs_to_tags[i];
		}
	}
	delete[] cws_pocs_to_tags;

	delete cws_dat;

	if(cws_model!=NULL) delete cws_model;

	delete tagging_decoder;
	if(tagging_model != NULL){
		for(int i = 0; i < tagging_model->l_size; i ++){
			if(tagging_label_info) delete[](tagging_label_info[i]);
		}
	}   
	delete[] tagging_label_info;
     
	if(tagging_pocs_to_tags){
		for(int i = 1; i < 16; i ++){
			delete[] tagging_pocs_to_tags[i];
		}
	}
	delete[] tagging_pocs_to_tags;

	delete tagging_dat;
	if(tagging_model!=NULL) delete tagging_model;
}

extern "C" char *getResult() {
	return result;
}

extern "C" void freeResult() {
	if (result != NULL) {
		std::free(result);
		result = NULL;
	}
}

extern "C" int seg(const char *in) {
	POCGraph poc_cands;
	POCGraph new_poc_cands;
	int rtn=1;
	thulac::RawSentence oriRaw;
	thulac::RawSentence raw;
	thulac::RawSentence tRaw;
	thulac::SegmentedSentence segged;
	thulac::TaggedSentence tagged;

	const int BYTES_LEN=10000;
	char* s=new char[ BYTES_LEN];
	char* out=new char[BYTES_LEN];
	std::string ori;
	bool isFirstLine = true;
	int codetype = -1;

	Chinese_Charset_Conv conv;

	std::ostringstream ost;
	clock_t start = clock();
	std::string str(in);
	std::istringstream ins(str);
	std::ostringstream ous;


	bool containsT = false;
	while(std::getline(ins,ori)){
		containsT = false;

		if(ori.length()>9999){
			ori = ori.substr(0,9999);
		}
		strcpy(s,ori.c_str());
		size_t in_left=ori.length();

		if(isFirstLine){ 
			size_t out_left=BYTES_LEN;
			codetype = conv.conv(s,in_left,out,out_left);
			if(codetype >=0){
				int outlen=BYTES_LEN - out_left;
				thulac::get_raw(oriRaw,out,outlen);
			}else{
				return int(false);
			}
			isFirstLine = false;
		}else{
			ous << " ";
			if(codetype == 0){
				thulac::get_raw(oriRaw,s,in_left);
			}else{
				size_t out_left=BYTES_LEN;
				codetype = conv.conv(s,in_left,out,out_left,codetype);
				int outlen=BYTES_LEN - out_left;
				thulac::get_raw(oriRaw,out,outlen);
			}
		}
		if(preprocesser->containsT(oriRaw)){
			preprocesser->clean(oriRaw,tRaw,poc_cands);
			preprocesser->T2S(tRaw, raw);
			containsT = true;
		}else{
			preprocesser->clean(oriRaw,raw,poc_cands);
		}

		if(raw.size()){
			cws_decoder->segment(raw, poc_cands, new_poc_cands);
			if(seg_only){
				cws_decoder->segment(raw, poc_cands, new_poc_cands);
				cws_decoder->get_seg_result(segged);
				ns_dict->adjust(segged);
				idiom_dict->adjust(segged);
				nz_dict->adjust(segged);
				noun_dict->adjust(segged);
				if(user_dict){
					user_dict->adjust(segged);
				}
				punctuation->adjust(segged);
				timeword->adjust(segged);
				if(useFilter){
					filter->adjust(segged);
				}
				if(codetype==0){
					for(int j = 0; j < segged.size(); j++){
						if(j!=0) ous<<" ";
						ous<<segged[j];
					}
				}else{
					for(int j = 0; j < segged.size(); j++){
						if(j!=0) ost<<" ";
						ost<<segged[j];
					}
					std::string str=ost.str();
					strcpy(s,str.c_str());
					size_t in_left=str.size();
					size_t out_left=BYTES_LEN;
					codetype = conv.invert_conv(s,in_left,out,out_left,codetype);
					int outlen=BYTES_LEN - out_left;
					ous<<std::string(out,outlen);
					ost.str("");
				}
			}else{

				if(use_second){
					Lattice lattice;
					hypergraph::Graph graph;
					tagging_decoder->segment(raw, new_poc_cands, lattice);
					hypergraph::lattice_to_graph(lattice, graph);
					decoder.decode(graph);
					hypergraph::graph_to_lattice(graph,lattice,1);
					lattice_to_sentence(lattice,tagged, (char)separator);
				}else{
					tagging_decoder->segment(raw, new_poc_cands, tagged);
					//tagging_decoder->segment(raw, poc_cands, tagged);
				}

				ns_dict->adjust(tagged);
				idiom_dict->adjust(tagged);
				nz_dict->adjust(tagged);
				ni_dict->adjust(tagged);
				noun_dict->adjust(tagged);
				adj_dict->adjust(tagged);
				verb_dict->adjust(tagged);
				// vm_dict->adjustSame(tagged);
				y_dict->adjustSame(tagged);

				if(user_dict){
					user_dict->adjust(tagged);
				}
				punctuation->adjust(tagged);
				timeword->adjustDouble(tagged);
				negword->adjust(tagged);
				verbword->adjust(tagged);
				if(useFilter){
					filter->adjust(tagged);
				}
				
				if(containsT && !useT2S){
					preprocesser->S2T(tagged, tRaw);
				}
				
				if(codetype==0){
					ous<<tagged;
				}else{
					ost<<tagged;
					std::string str=ost.str();
					strcpy(s,str.c_str());
					size_t in_left=str.size();
					size_t out_left=BYTES_LEN;
					codetype = conv.invert_conv(s,in_left,out,out_left,codetype);
					int outlen=BYTES_LEN - out_left;
					ous<<std::string(out,outlen);
					ost.str("");
				}
			}
		}
	}
	
	clock_t end = clock();
	double duration = (double)(end - start) / CLOCKS_PER_SEC;
	delete [] s;
	delete [] out;
	std::string ostr = ous.str();
	size_t len = ostr.length();
	if (len > result_size) {
		len = result_size;
	}
	char * p = (char *)calloc(len+1, 1);
	if (p == NULL) {
		return int(false);
	}
	std::memcpy(p, ostr.c_str(), ostr.length());
	result = p;
	return 1;
}
