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
/// \file B4/B4a/src/PrimaryGeneratorAction.cc
/// \brief Implementation of the B4::PrimaryGeneratorAction class
#include "PrimaryGeneratorAction.hh"
#include <thread>
#include <chrono>

#include <Randomize.hh>
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4Box.hh"

// Socket Imports
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <mutex>
#include <queue>

namespace B4
{

// --- SHARED DATA ---
struct ParticleCommand {
    std::string particleType;
    G4double energy;
};

static std::queue<ParticleCommand> g_commandQueue;
static std::mutex g_queueMutex;
static bool g_networkThreadStarted = false; // Ensure we only start one listener

// --- THE NETWORK LISTENER (Runs in background) ---
void NetworkListenerLoop() {
    G4cout << "🎧 NETWORK LISTENER THREAD STARTED on Port 5003" << G4endl;

    // 1. Setup Server Socket
    int serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(5003);

    if (bind(serverSocket, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        G4cerr << "❌ Listener failed to bind port 5003" << G4endl;
        return;
    }
    listen(serverSocket, 1);

    // 2. Main Loop
    while (true) {
        // Accept Client
        struct sockaddr_in clientAddr;
        socklen_t len = sizeof(clientAddr);
        int clientSocket = accept(serverSocket, (struct sockaddr*)&clientAddr, &len);

        if (clientSocket >= 0) {
            G4cout << "🔗 Client Connected!" << G4endl;
            char buffer[1024];

            // Read Loop for this client
            while (true) {
                int bytes = recv(clientSocket, buffer, 1023, 0);
                if (bytes <= 0) break; // Client disconnected

                buffer[bytes] = '\0';
                std::string data(buffer);

                // Parse "GENERATE,e-,1000" or "BATCH,pi-,500,10"
                std::istringstream stream(data);
                std::string line;
                while(std::getline(stream, line)) {
                    // Simple Parsing Logic
                    std::stringstream ss(line);
                    std::string segment;
                    std::vector<std::string> parts;
                    while(std::getline(ss, segment, ',')) parts.push_back(segment);

                    if (parts.size() >= 3) {
                        ParticleCommand cmd;
                        cmd.particleType = parts[1];
                        cmd.energy = std::stod(parts[2]) * MeV;
                        int count = (parts[0] == "BATCH" && parts.size() > 3) ? std::stoi(parts[3]) : 1;

                        // PUSH TO QUEUE (Thread Safe)
                        {
                            std::lock_guard<std::mutex> lock(g_queueMutex);
                            for(int i=0; i<count; i++) {
                                g_commandQueue.push(cmd);
                            }
                        }
                        G4cout << "📥 Buffered " << count << " events." << G4endl;
                    }
                }
            }
            close(clientSocket);
            G4cout << "🔌 Client Disconnected" << G4endl;
        }
    }
}

// --- CLASS IMPLEMENTATION ---

PrimaryGeneratorAction::PrimaryGeneratorAction() {
    fParticleGun = new G4ParticleGun(1);

    // Start the Network Listener ONCE (detached thread)
    // This allows Geant4 workers to focus 100% on physics
    static std::once_flag flag;
    std::call_once(flag, [](){
        std::thread t(NetworkListenerLoop);
        t.detach(); // Let it run in the background forever
    });
}

PrimaryGeneratorAction::~PrimaryGeneratorAction() {
    delete fParticleGun;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event) {
    ParticleCommand cmd;
    bool hasData = false;

    // --- CONSUMER LOGIC ---
    // Workers sit here waiting for work.
    // They don't care about sockets, ports, or parsing.
    // They just want a command struct.
    while (!hasData) {
        {
            std::lock_guard<std::mutex> lock(g_queueMutex);
            if (!g_commandQueue.empty()) {
                cmd = g_commandQueue.front();
                g_commandQueue.pop();
                hasData = true;
            }
        }

        if (!hasData) {
            // Sleep briefly to avoid burning CPU while waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    // --- EXECUTE (Parallel Physics Starts Here) ---
    // Once we leave this function, Geant4 Kernel takes over
    // and runs the simulation on this thread in parallel.

    auto particle = G4ParticleTable::GetParticleTable()->FindParticle(cmd.particleType);
    if (!particle) particle = G4ParticleTable::GetParticleTable()->FindParticle("e-");

    fParticleGun->SetParticleDefinition(particle);
    fParticleGun->SetParticleEnergy(cmd.energy);

    fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0., 0., 1.));

    // Standard Position Logic
    auto worldLV = G4LogicalVolumeStore::GetInstance()->GetVolume("World");
    G4Box* worldBox = dynamic_cast<G4Box*>(worldLV->GetSolid());
    G4double zPos = worldBox ? -worldBox->GetZHalfLength() : 0;

    fParticleGun->SetParticlePosition(G4ThreeVector(0., 0., zPos));
    fParticleGun->GeneratePrimaryVertex(event);
}

} // namespace B4