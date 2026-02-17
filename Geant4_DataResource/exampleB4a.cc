//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
//
/// \file exampleB4a.cc
/// \brief Main program of the B4a example

#include <queue>

#include "CommandQueue.hh"
#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"
#include "FTFP_BERT.hh"

#include "G4RunManagerFactory.hh"
#include "G4SteppingVerbose.hh"
#include "G4UIExecutive.hh"
#include "G4UIcommand.hh"
#include "G4UImanager.hh"
#include "G4VisExecutive.hh"
// #include "Randomize.hh"


#include <thread>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <vector>

#include <thread>
#include <chrono>

#include <Randomize.hh>
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4Box.hh"
#include "include/CommandQueue.hh"
//#include "include/CommandQueue.hh"
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

namespace
{
void PrintUsage()
{
  std::cout << " Usage: " << std::endl;
  std::cout << " exampleB4a [-m macro ] [-u UIsession] [-t nThreads] [-vDefault]" << std::endl;
  std::cout << "   note: -t option is available only for multi-threaded mode." << std::endl;
}
}  // namespace

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......


namespace B4 {
  std::queue<ParticleCommand> g_commandQueue;
  std::mutex g_queueMutex;
}

// --- THE LISTENER LOOP ---
void NetworkListenerLoop() {
  std::cout << "🎧 NETWORK LISTENER THREAD STARTED on Port 5003" << std::endl;

  int serverSocket = socket(AF_INET, SOCK_STREAM, 0);

  int opt = 1;
  setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  struct sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(5003);

  if (bind(serverSocket, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
std::cout<< "❌ Listener failed to bind port 5003. Error: " << std::endl;    return;
  }
  listen(serverSocket, 1);

  while (true) {
    struct sockaddr_in clientAddr;
    socklen_t len = sizeof(clientAddr);
    int clientSocket = accept(serverSocket, (struct sockaddr*)&clientAddr, &len);

    if (clientSocket >= 0) {
      char buffer[1024];
      std::cout << "🔗 Java Controller Connected!" << std::endl;
      while (true){
      std::cout << "📩 COMMAND RECEIVED! Processing..." << std::endl;

      int bytes = recv(clientSocket, buffer, 1023, 0);
      if (bytes <= 0) {
        std::cout << "🔌 Controller Disconnected." << std::endl;
        close(clientSocket);
        break;
      }
      if (bytes > 0) {
        buffer[bytes] = '\0';
        std::string data(buffer);

        // --- PARSING LOGIC RESTORED ---
        std::stringstream ss(data); // e.g., "GENERATE,e-,1000"
        std::string segment;
        std::vector<std::string> parts;

        while(std::getline(ss, segment, ',')) {
          parts.push_back(segment);
        }

        if (parts.size() >= 3) {
          B4::ParticleCommand cmd;
          cmd.particleType = parts[1]; // e.g. "e-"

          // Safe conversion for energy
          try {
            cmd.energy = std::stod(parts[2]) * MeV;
          } catch (...) {
            cmd.energy = 300 * MeV; // Default if parse fails
          }

          // Handle Count (Batch size)
          int count = 1;
          if (parts.size() > 3) {
            try { count = std::stoi(parts[3]); } catch(...) {}
          }

          // Push to Queue
          {
            std::lock_guard<std::mutex> lock(B4::g_queueMutex);
            for(int i=0; i<count; i++) {
              B4::g_commandQueue.push(cmd);
            }
          }
          std::cout << "📥 Queued " << count << " particles of type " << cmd.particleType << std::endl;
           }
        }
      }
      close(clientSocket);
    }
  }
}


int main(int argc, char** argv)
{
  // Evaluate arguments
  //
  if (argc > 7) {
    PrintUsage();
    return 1;
  }

  G4String macro;
  G4String session;
  G4bool verboseBestUnits = true;
#ifdef G4MULTITHREADED
  G4int nThreads = 0;
#endif
  for (G4int i = 1; i < argc; i = i + 2) {
    if (G4String(argv[i]) == "-m")
      macro = argv[i + 1];
    else if (G4String(argv[i]) == "-u")
      session = argv[i + 1];
#ifdef G4MULTITHREADED
    else if (G4String(argv[i]) == "-t") {
      nThreads = G4UIcommand::ConvertToInt(argv[i + 1]);
    }
#endif
    else if (G4String(argv[i]) == "-vDefault") {
      verboseBestUnits = false;
      --i;  // this option is not followed with a parameter
    }
    else {
      PrintUsage();
      return 1;
    }
  }

  // Detect interactive mode (if no macro provided) and define UI session
  //
  G4UIExecutive* ui = nullptr;
  if (!macro.size()) {
    ui = new G4UIExecutive(argc, argv, session);
  }

  // Optionally: choose a different Random engine...
  // G4Random::setTheEngine(new CLHEP::MTwistEngine);

  // Use G4SteppingVerboseWithUnits
  if (verboseBestUnits) {
    G4int precision = 4;
    G4SteppingVerbose::UseBestUnit(precision);
  }

  // Construct the default run manager
  //

  // 1. START LISTENER FIRST (Before anything else)
  std::thread networkThread(NetworkListenerLoop);
  networkThread.detach();


  auto runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default);
#ifdef G4MULTITHREADED
  if (nThreads > 0) {
    runManager->SetNumberOfThreads(nThreads);
  }
#endif

  // Set mandatory initialization classes
  //
  auto detConstruction = new B4::DetectorConstruction();
  runManager->SetUserInitialization(detConstruction);

  auto physicsList = new FTFP_BERT;
  runManager->SetUserInitialization(physicsList);

  auto actionInitialization = new B4a::ActionInitialization(detConstruction);
  runManager->SetUserInitialization(actionInitialization);

  // Initialize visualization
  //
  auto visManager = new G4VisExecutive;
  // G4VisExecutive can take a verbosity argument - see /vis/verbose guidance.
  // auto visManager = new G4VisExecutive("Quiet");
  visManager->Initialize();

  // Get the pointer to the User Interface manager
  auto UImanager = G4UImanager::GetUIpointer();

  // Process macro or start UI session
  //
  if (macro.size()) {
    // batch mode
    G4String command = "/control/execute ";
    UImanager->ApplyCommand(command + macro);
  }
  else {
    // interactive mode : define UI session
    UImanager->ApplyCommand("/control/execute init_vis.mac");
    UImanager->ApplyCommand("/run/beamOn 10000");
    if (ui->IsGUI()) {
      UImanager->ApplyCommand("/control/execute gui.mac");
    }
    ui->SessionStart();
    delete ui;
  }

  // Job termination
  // Free the store: user actions, physics_list and detector_description are
  // owned and deleted by the run manager, so they should not be deleted
  // in the main() program !

  delete visManager;
  delete runManager;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....
