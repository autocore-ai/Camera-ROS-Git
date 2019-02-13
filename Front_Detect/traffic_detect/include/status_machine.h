#ifndef STATUS_MACHINE_H_H_H
#define STATUS_MACHINE_H_H_H
#include<string.h>
#include<map>
struct TFSCODE
{
    bool codes[6];
   std::map<std::string,int> hash_map;
    
};

class TFSMachine
{
    public:
        //----
        TFSMachine()
        {
            // default status: stop_up,stop_left,go_right
            init_tls_code();
        }
        bool add_round_tl_signal(bool is_green=false); //
        void get_tls_go_code(bool &go_up,bool &go_left,bool &go_right);
        bool add_arrow_tl_signal(std::string tls_key);
        std::string get_status_label();
    private:
        //
        TFSCODE m_tl_status;
        void init_tls_code();
        int get_tls_code_idx(std::string tls_key);
        bool update_single_signal(std::string tls_key);

};

class TFSMaintainer
{
public:
	// refresh_epoch: refresh maintainer with initialized status every m_maintain_time frames.
	// default status: green-go 
	TFSMaintainer( int refresh_epoch = 5) 
	{
		if(1>refresh_epoch)
			refresh_epoch = 1;
		m_maintain_time = refresh_epoch;
		m_go_up = true;
		m_go_left = true;
		m_go_right = true;
		m_stop_up_clock =0;
		m_stop_left_clock =0;
		m_stop_left_clock =0;	
	}

	void get_stable_go_status(bool &go_up,bool &go_left, bool &go_right)
	{
		go_up = m_go_up;
		go_left = m_go_left;
		go_right = m_go_right;
	}
	void update_status(bool go_up,bool go_left,bool go_right);
        std::string get_stable_status_label();
private:
	int m_maintain_time;
	bool m_go_up ;
	bool m_go_left ;	
	bool m_go_right;
	int  m_stop_up_clock;
	int  m_stop_left_clock;
	int  m_stop_right_clock;
};
#endif
