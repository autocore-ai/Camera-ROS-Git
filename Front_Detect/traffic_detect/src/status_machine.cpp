#include<iostream>
#include<status_machine.h>
using namespace std;
// init status code
void TFSMachine::init_tls_code()
{
    m_tl_status.classname2idx[string("goUp")]=0;
    m_tl_status.classname2idx[string("goLeft")]=1;
    m_tl_status.classname2idx[string("goRight")]=2;
    m_tl_status.classname2idx[string("stopUp")]=3;
    m_tl_status.classname2idx[string("stopLeft")]=4;
    m_tl_status.classname2idx[string("stopRight")]=5;
    /*
    ----------- default status ----------------
    go_up:0    stop_up:1 \
    go_left:0  stop_left:1 \
    go_right:1 stop_right:0
    --------------------------------------------
    */
    update_single_signal("goUp");
    update_single_signal("goLeft");
    update_single_signal("goRight");    
}

// get traffic light index 
int TFSMachine::get_tls_code_idx(string tls_key)
{
    if(m_tl_status.classname2idx.find(tls_key)!=m_tl_status.classname2idx.end())
    {
        return m_tl_status.classname2idx[tls_key];
    }
    else
        return -1;
}

// update single traffic light status
bool TFSMachine::update_single_signal(string tls_key)
{
    int idx = get_tls_code_idx(tls_key);
    if(idx<0|| idx>5)
    {
        std::cout<<"Erroring: unknown traffic light singal:("<<tls_key<<")\n";
        return false;
    }
    else //update status pair
    {
        m_tl_status.codes[idx]=true;
        if(idx>=3) //update stop groups ones,go groups zeros
        {
            m_tl_status.codes[idx-3] =false;
        }
        else //update go groups ones, stop group zeros
        {
            m_tl_status.codes[idx + 3] =true;
        }
    }        
    return true;
}

// get traffic light signals go groups status code
void TFSMachine::get_tls_go_code(bool &go_up,bool &go_left,bool &go_right)
{
    go_up =m_tl_status.codes[0];
    go_left = m_tl_status.codes[1];
    go_right = m_tl_status.codes[2];
}

// add round traffic light signal,
// go_right is default active under red round blob
bool TFSMachine::add_round_tl_signal(bool is_green/*= false*/)
{
    if(is_green)// open go group signals
    {
        update_single_signal(string("goUp"));
        update_single_signal(string("goLeft"));
        update_single_signal(string("goRight"));
    }
    else //open stop goup signals
    {
        update_single_signal(string("stopUp"));
        update_single_signal(string("stopLeft"));
        update_single_signal(string("goRight"));
    }
    return true;
}
bool TFSMachine::add_arrow_tl_signal(string tls_key)
{
    return update_single_signal( tls_key);
}

std::string TFSMachine::get_status_label()
{   bool go_up,go_left,go_right;
    string label="GO:";
    get_tls_go_code(go_up,go_left,go_right);
    if(go_left)
        label = label + "--left";
    if(go_up)
        label = label + "--up";
    if(go_right)
        label = label + "--right";
    return label;
}

// status maintainer update traffic light status
void TFSMaintainer::update_status(bool go_up,bool go_left,bool go_right)
{
	// stop_clock --;
	if(m_stop_up_clock>0)
		m_stop_up_clock -=1;
	if(m_stop_left_clock>0)
		m_stop_left_clock -=1;
	if(m_stop_right_clock > 0)
		m_stop_right_clock -=1;

	// when meet stop up/left/right signals
	// restart stop clock
	if(!go_up)
	{
		m_go_up = go_up;
		m_stop_up_clock = m_maintain_time;			
	}
	if(!go_left)
	{
		m_go_left = go_left;
		m_stop_left_clock = m_maintain_time;
	}
	if(!go_right)
	{
		m_go_right = go_right;
		m_stop_right_clock = m_maintain_time;
	}
	
	// when stop clock died, refresh maintainer with default status
	if(0==m_stop_up_clock)
		m_go_up = true;
	if(0==m_stop_left_clock)
		m_go_left = true;
	if(0==m_stop_right_clock)
		m_go_right = true;

} 	

std::string TFSMaintainer::get_stable_status_label()
{   bool go_up,go_left,go_right;
    std::string label;
    get_stable_go_status(go_up,go_left,go_right);
    if(go_up)
        label =label+"Up:Y ";
    else
        label =label+"Up:N ";
    if(go_left)
        label =label+"Left:Y ";
    else 
        label =label+"Left:N ";
    if(go_right)
        label =label+"Right:Y ";
    else
        label =label+"Right:N ";
    return label;
}

