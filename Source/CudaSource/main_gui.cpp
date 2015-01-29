#include <cstdio>
#include <stdio.h>
#include "imgIO.hpp"
#include "DCIDetect.h"
#include <vector>
#include <iostream>
#include <wx/wxprec.h>
#include "helper_timer.h"
#include "HaarAlgorithm.h"
#include "IntegralImage.h"
#ifndef WX_PRECOMP
    #include <wx/wx.h>
#endif
void mark2(HaarRectangle2& rect, unsigned char* imgGrey, UInt stride)
{
	for (int i = 0; i < rect.height; ++i)
	{
		imgGrey[(rect.top + i)*stride + rect.left] = 250;
		imgGrey[(rect.top + i)*stride + rect.left + rect.width] = 250;
	}
	for (int i = 0; i < rect.width; ++i)
	{
		imgGrey[(rect.top*stride) + rect.left + i] = 250; // gora
		imgGrey[((rect.top + rect.height)*stride) + rect.left + i] = 250; //dol
	}
}


void mark(HaarRectangle& rect, unsigned char* imgGrey, UInt stride)
{
	for (UInt i = 0; i < rect.height; ++i)
	{
		imgGrey[(rect.top + i)*stride + rect.left] = 250;
		imgGrey[(rect.top + i)*stride + rect.left + rect.width] = 250;
	}
	for (UInt i = 0; i < rect.width; ++i)
	{
		imgGrey[(rect.top*stride) + rect.left + i] = 250; // gora
		imgGrey[((rect.top + rect.height)*stride) + rect.left + i] = 250; //dol
	}
}


class MyApp: public wxApp
{
public:
    virtual bool OnInit();
};
class MyFrame: public wxFrame
{
public:
    MyFrame(const wxString& title, const wxPoint& pos, const wxSize& size);
private:
    void OnHello(wxCommandEvent& event);
	void OpenPhoto(wxCommandEvent& event);
    void OnExit(wxCommandEvent& event);
    void OnAbout(wxCommandEvent& event);
	void OnButtonClick(wxCommandEvent& event);
    wxDECLARE_EVENT_TABLE();
	wxImage photo;
	wxButton *button;
    wxPanel* pane;
	wxRadioBox *radiobox;
	wxRadioButton *HaarRadio;
	wxRadioButton *SkinRadio;
	bool picture_loaded;
	 char* pathIn;
	 char* pathOutGrey;
	 char* pathOutColor;

	 unsigned char *imgColor;
	 unsigned char *imgGrey;
	 unsigned char *imgClean;

	DCIDetect *detect;
};
enum
{
    ID_Hello = 1,
	BUTTON_Hello = 2
};

//BUTTON_Hello = wxID_HIGHEST + 1;
wxBEGIN_EVENT_TABLE(MyFrame, wxFrame)
    EVT_MENU(ID_Hello,   MyFrame::OnHello)
	EVT_MENU(wxID_OPEN, MyFrame::OpenPhoto)
    EVT_MENU(wxID_EXIT,  MyFrame::OnExit)
    EVT_MENU(wxID_ABOUT, MyFrame::OnAbout)
	EVT_BUTTON(BUTTON_Hello, MyFrame::OnButtonClick)

	//EVT_PAINT(wxImagePanel::paintEvent)
wxEND_EVENT_TABLE()
wxIMPLEMENT_APP(MyApp);
bool MyApp::OnInit()
{
    MyFrame *frame = new MyFrame( "Face Detector", wxPoint(50, 50), wxSize(450, 350) );
    frame->Show( true );
    return true;
}
MyFrame::MyFrame(const wxString& title, const wxPoint& pos, const wxSize& size)
        : wxFrame(NULL, wxID_ANY, title, pos, size)
{
	pane = new wxPanel(this, wxID_ANY);
    wxMenu *menuFile = new wxMenu;
	menuFile->Append(wxID_OPEN, "Wczytaj zdjęcie\tCtrl-O");
    menuFile->AppendSeparator();
    menuFile->Append(wxID_EXIT, "Zakończ...\tCtrl-Z");
    wxMenu *menuHelp = new wxMenu;
    menuHelp->Append(wxID_ABOUT, "O autorach\tCtrl-H");
    wxMenuBar *menuBar = new wxMenuBar;
    menuBar->Append( menuFile, "&Plik" );
    menuBar->Append( menuHelp, "&Help" );
    SetMenuBar( menuBar );
    CreateStatusBar();
    SetStatusText( "Wybierz opcję z menu Plik" );
	wxInitAllImageHandlers();
	picture_loaded = false;
	button = new wxButton(pane, BUTTON_Hello, wxT("Szukaj twarzy!"), wxPoint(10, GetSize().GetHeight() - 120));
	//+radiobox->
HaarRadio = new wxRadioButton(pane, -1, 
      wxT("Algorytm obszarów kontrastowych"), wxPoint(110, GetSize().GetHeight() - 120), wxDefaultSize, wxRB_GROUP);

SkinRadio = new wxRadioButton(pane, -1, 
	 wxT("Algorytm szukający kolorów i tekstur"), wxPoint(110, GetSize().GetHeight() - 100));
imgColor = NULL;
imgGrey = NULL;
imgClean = NULL;
}
void MyFrame::OnExit(wxCommandEvent& event)
{
    Close( true );
}
void MyFrame::OnAbout(wxCommandEvent& event)
{
    wxMessageBox( "Autorami aplikacji są:\nMichał Kosyl\nPaweł Soćko\nLeszek Tatara",
                  "Autorzy programu", wxOK | wxICON_ERROR );
}
void MyFrame::OnHello(wxCommandEvent& event)
{
    wxLogMessage("Hello world from wxWidgets!");
}

void MyFrame::OnButtonClick(wxCommandEvent& event) {
	if (!picture_loaded) {
		    wxMessageBox( "Użytkownik nie wczytał żadnego zdjęcia.\nProszę skorzystać w tym celu z menu plik.", "Zonk", wxOK | wxICON_INFORMATION );
	} else {
	if (SkinRadio->GetValue()) {
	SetStatusText("Szukam twarzy...");
	ImgIO imgIO;
	imgIO.ReadImgColor((const char*)pathIn, imgColor);
	imgIO.ColorToGray(imgColor, imgGrey);
	imgIO.ColorToGray(imgColor, imgClean);

		 int imgSizeX = imgIO.getSizeX();
	 int imgSizeY = imgIO.getSizeY();
	 detect = new DCIDetect();
		std::vector<HaarRectangle2> result = detect->Run(imgColor, imgGrey, imgSizeX, imgSizeY);

	for (std::vector<HaarRectangle2>::iterator i = result.begin(); i != result.end(); ++i)
	{
		mark2(*i, imgClean, imgIO.getSizeX());
	}
		const char* pathGrey = "obrazTestOutGrey.jpg";

	imgIO.WriteImgGrey(pathGrey, imgClean);
		wxString photo_file;
	wxClientDC dc(pane);
	photo_file = pathGrey;

	dc.Clear();
	if (photo_file.find(wxString(".jpg")) != wxNOT_FOUND)
		photo.LoadFile(photo_file, "image/jpeg");
	else
	if (photo_file.find(wxString(".png")) != wxNOT_FOUND)
		photo.LoadFile(photo_file, "image/png");
	else
	if (photo_file.find(wxString(".gif")) != wxNOT_FOUND)
		photo.LoadFile(photo_file, "image/gif");
		//dc.DrawBitmap(photo,10,10,false);
		dc.DrawBitmap(photo, (GetClientSize().GetX() - photo.GetWidth()) / 2, 10, false);

	SetStatusText("Zakończono operację.");
	} else { ImgIO imgIO;
	imgIO.ReadImgColor((const char*)pathIn, imgColor);
	imgIO.ColorToGray(imgColor, imgGrey);
	imgIO.ColorToGray(imgColor, imgClean);

		 int imgSizeX = imgIO.getSizeX();
	 int imgSizeY = imgIO.getSizeY();
	HaarAlgorithm alg;

//	timer(&start);
	std::vector<HaarRectangle> result = alg.execute(imgIO.getSizeX(), imgIO.getSizeY(), imgGrey);
	for (std::vector<HaarRectangle>::iterator i = result.begin(); i != result.end(); ++i)
	{
		mark(*i, imgGrey, imgIO.getSizeX());
	}
		const char* pathGrey = "obrazTestOutGrey.jpg";

	imgIO.WriteImgGrey(pathGrey, imgClean);
		wxString photo_file;
	wxClientDC dc(pane);
	photo_file = pathGrey;

	dc.Clear();
	if (photo_file.find(wxString(".jpg")) != wxNOT_FOUND)
		photo.LoadFile(photo_file, "image/jpeg");
	else
	if (photo_file.find(wxString(".png")) != wxNOT_FOUND)
		photo.LoadFile(photo_file, "image/png");
	else
	if (photo_file.find(wxString(".gif")) != wxNOT_FOUND)
		photo.LoadFile(photo_file, "image/gif");
		//dc.DrawBitmap(photo,10,10,false);
		dc.DrawBitmap(photo, (GetClientSize().GetX() - photo.GetWidth()) / 2, 10, false);

	SetStatusText("Zakończono operację.");
}

	}
	}

void MyFrame::OpenPhoto(wxCommandEvent& event) {

	wxString photo_file;
	wxClientDC dc(pane);
	wxFileDialog wxfd(this, ("Wybierz fotografię"), "", "", "Pliki (*.gif, *.jpg, *.png)|*jpg;*png;*gif", wxFD_OPEN | wxFD_FILE_MUST_EXIST);
	if (wxfd.ShowModal() == wxID_OK) {
	//SetStatusText(wxfd.GetPath());
	photo_file = wxfd.GetPath();
	
	//photo.Clear();
	dc.Clear();
	if (photo_file.find(wxString(".jpg")) != wxNOT_FOUND)
		photo.LoadFile(photo_file, "image/jpeg");
	else
	if (photo_file.find(wxString(".png")) != wxNOT_FOUND)
		photo.LoadFile(photo_file, "image/png");
	else
	if (photo_file.find(wxString(".gif")) != wxNOT_FOUND)
		photo.LoadFile(photo_file, "image/gif");
//	photo_area.DrawBitmap(photo,0,0, false);
	///wxFilterInputStream picture
	photo.GetHeight();
	photo.GetWidth();
	//this->set
	if (photo.GetWidth() >= 380)
		SetClientSize(photo.GetWidth()+20, photo.GetHeight()+60);
	else
		SetClientSize(400, photo.GetHeight()+60);

	button->SetPosition(wxPoint(10, photo.GetHeight()+10));
	HaarRadio->SetPosition(wxPoint(110, photo.GetHeight()+10));
	SkinRadio->SetPosition(wxPoint(110, photo.GetHeight()+35));
	
	dc.DrawBitmap(photo, (GetClientSize().GetX() - photo.GetWidth()) / 2, 10, false);
	//dc.DrawBitmap(photo,10,10,false);
	std::string pathIns = photo_file.ToStdString();
	pathIn = new char[pathIns.length()+1];
	strcpy(pathIn, pathIns.c_str());
	pathIn[pathIns.length()] = '\0';
	picture_loaded = true;
	imgColor = NULL;
	imgGrey = NULL;
	imgClean = NULL;
	//Fit();
	//SetStatusText(	photo.GetWidth());
	}
}
